import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter


def train(train_dloader_list, model_list, optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, tau=0.6, mix_aug=True, get_KL_values=False, get_pseudolabel_acc=False):
    scale_names=['final', 'scale1', 'scale2', 'scale4']
    task_criterion = nn.CrossEntropyLoss().cuda()
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        # Train model locally on source domains
        for train_dloader, model, optimizer in zip(train_dloader_list[1:],
                                                    model_list[1:],
                                                    optimizer_list[1:]):

            # check if the source domain is the malicious domain with poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            for i, (image_s, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break
                image_s = image_s.cuda()
                label_s = label_s.long().cuda()
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                # reset grad
                optimizer.zero_grad()
                # each source domain do optimize
                output_s = model(image_s)
                task_loss_s = get_multiscale_classification_loss(output_s, label_s, task_criterion)
                task_loss_s.backward()
                optimizer.step()
    # Domain adaptation on target domain
    confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    # We use I(n_i>=1)/(N_T) to adjust the weight for knowledge distillation domain
    target_weight = [0, 0]
    consensus_focus_dict = {}
    temporal_consistency_dict = {}
    if get_KL_values:
        epoch_kl_per_scale = {
            'final': [],
            'scale1': [],
            'scale2': [],
            'scale4': []
        }
    if get_pseudolabel_acc:
        pseudolabel_accuracy_tracker = {
            'correct': 0,
            'total': 0,
            'per_scale': {'final': {'correct': 0, 'total': 0},
                        'scale1': {'correct': 0, 'total': 0},
                        'scale2': {'correct': 0, 'total': 0},
                        'scale4': {'correct': 0, 'total': 0}}
        }

    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0
    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        if i >= batch_per_epoch:
            break
        optimizer_list[0].zero_grad()
        image_t = image_t.cuda()
        # Knowledge Vote
        with torch.no_grad():
            # knowledge_list: [B, source_domain_num*num_scales, num_classes]
            knowledge_list = [torch.cat([torch.softmax(model_list[i](image_t)[k], dim=1).unsqueeze(1) 
                              for k in scale_names], dim=1)
                              for i in range(1, source_domain_num + 1)]
            knowledge_list = torch.cat(knowledge_list, 1)
        _, consensus_knowledge, consensus_weight, consensus_mask = knowledge_vote_multiscale(knowledge_list, confidence_gate,
                                                                  num_classes=num_classes)
        # Target weight still considers the percentage of confident pseudolabels. Though this contradicts method in paper
        target_weight[0] += torch.sum(consensus_weight).item()
        target_weight[1] += consensus_weight.size(0)

        # Reshape consensus_knowledge and consensus_weight 
        # from [B, source_domain_num*num_scales, num_classes] to [B, source_domain_num, num_scales, num_classes]
        B = consensus_knowledge.size(0)
        M = source_domain_num
        S = 4
        # [B, N, C] → [B, M, S, C]
        consensus_knowledge = consensus_knowledge.view(B, M, S, -1)
        # [B, N] → [B, M, S]
        consensus_mask = consensus_mask.view(B, M, S)

        # DEBUG: Calculate pseudolabel accuracy per scale (if pseudolabels are confident)
        if get_pseudolabel_acc:
            # Get pseudolabel predictions (hard labels) without mixup
            weighted_sum_original = torch.sum(consensus_knowledge * consensus_mask[..., None], dim=1)  # [B, S, C]
            weight_sum_original = torch.sum(consensus_mask, dim=1, keepdim=False)  # [B, S]
            consensus_per_scale_original = weighted_sum_original / (weight_sum_original[..., None] + 1e-6)  # [B, S, C]

            # Get hard pseudolabels from original consensus (no mixup)
            pseudolabel_per_scale = consensus_per_scale_original.argmax(dim=2)  # [B, S]
            
            # label_t shape: [B] - true labels
            label_t = label_t.long().cuda()
            true_labels = label_t.unsqueeze(1).expand(-1, S)  # [B, S] - repeat for all scales

            # Calculate per-scale accuracy
            scale_names_list = ['final', 'scale1', 'scale2', 'scale4']
            confident_pseudolabel_mask = consensus_weight.bool()
            for scale_idx, scale_name in enumerate(scale_names_list):
                # Filter by consensus weight (confident pseudolabels)
                confident_pseudolabels = pseudolabel_per_scale[confident_pseudolabel_mask, scale_idx]
                confident_true_labels = true_labels[confident_pseudolabel_mask, scale_idx]  
                correct_preds = (confident_pseudolabels == confident_true_labels).sum().item()
                pseudolabel_accuracy_tracker['per_scale'][scale_name]['correct'] += correct_preds
                pseudolabel_accuracy_tracker['per_scale'][scale_name]['total'] += confident_true_labels.size(0)
            
            # Calculate overall accuracy (average across scales)
            confident_pseudolabels_all = pseudolabel_per_scale[confident_pseudolabel_mask]
            confident_true_labels_all = true_labels[confident_pseudolabel_mask]
            correct_overall = (confident_pseudolabels_all == confident_true_labels_all).sum().item()
            pseudolabel_accuracy_tracker['correct'] += correct_overall
            pseudolabel_accuracy_tracker['total'] += confident_true_labels_all.nelement()

        # Perform data augmentation with mixup
        if mix_aug:
            lam = np.random.beta(2, 2)
        else:
            # Do not perform mixup
            lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        # mixed_consensus to output 4 pseudolabel distributions instead [B, M, S, C]
        mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
        # Aggregate over sources and normalise 
        weighted_sum = torch.sum(mixed_consensus * consensus_mask[..., None], dim=1)  # [B, S, C]
        weight_sum = torch.sum(consensus_mask, dim=1, keepdim=False)  # [B, S]
        consensus_per_scale = weighted_sum / (weight_sum[..., None] + 1e-6) # avoid zero division 
        # Calculate per scale KL divergence 
        output_t = model_list[0](mixed_image)
        student_log_probs = torch.stack([torch.log_softmax(output_t[k], dim=1)
                                         for k in scale_names], dim=1)
        kl_per_scale = torch.sum(-1* consensus_per_scale * student_log_probs, dim=2)
        # DEBUG: save kl_per_scale to check if it converges to 0 quickly
        if get_KL_values:
            for scale_idx, scale_name in enumerate(scale_names):
                # Extract KL values for this scale across the batch: [B]
                # Only consider samples with confident pseudolabels 
                kl_values = kl_per_scale[confident_pseudolabel_mask, scale_idx].detach().cpu().numpy()
                # Store all batch samples for this epoch
                epoch_kl_per_scale[scale_name].extend(kl_values)
        kl_per_sample = kl_per_scale.mean(dim=1)
        task_loss_t = torch.mean(consensus_weight * kl_per_sample)
        task_loss_t.backward()
        optimizer_list[0].step()
        # Calculate consensus focus 
        # TODO: verify if this is correct or if the multiscale version needs to be made 
        consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
                                                         source_domain_num, num_classes)
        # Calculate temporal consistency score
        temporal_consistency_dict = calculate_temporal_consistency(knowledge_list, source_domain_num, num_scales=4)

    # Consensus Focus Re-weighting
    target_parameter_alpha = target_weight[0] / target_weight[1]
    target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    epoch_domain_weight = []
    epoch_temporal_consistency_weight = []
    source_total_weight = 1 - target_weight
    for i in range(1, source_domain_num + 1):
        epoch_domain_weight.append(consensus_focus_dict[i])
        epoch_temporal_consistency_weight.append(temporal_consistency_dict[i])
    if sum(epoch_domain_weight) == 0:
        epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
    if sum(epoch_temporal_consistency_weight) == 0:
        epoch_temporal_consistency_weight = [v + 1e-3 for v in epoch_temporal_consistency_weight]
    # Combine consensus focus and temporal consistency for final domain weight. Tau is hyperparameter to balance the two factors.
    epoch_domain_weight = [round(source_total_weight * (tau * v / sum(epoch_domain_weight) + 
                                                        ((1-tau)* k / sum(epoch_temporal_consistency_weight))), 4) 
                           for v, k in zip(epoch_domain_weight, epoch_temporal_consistency_weight)]
    # target domain weight at index 0
    epoch_domain_weight.insert(0, target_weight)
    # Update domain weight with moving average
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)
    # Model aggregation and Batchnorm MMD
    federated_average(model_list, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # Recording domain weight in logs
    writer.add_scalar(tag="Train/target_domain_weight", scalar_value=target_weight, global_step=epoch + 1)
    for i in range(0, len(train_dloader_list) - 1):
        writer.add_scalar(tag="Train/source_domain_{}_weight".format(source_domains[i]),
                          scalar_value=domain_weight[i + 1], global_step=epoch + 1)
    print("Source Domains:{}, Domain Weight :{}".format(source_domains, domain_weight[1:]))
    if get_KL_values:
        epoch_kl_stats = {}
        for scale_name in scale_names:
            kl_values = np.array(epoch_kl_per_scale[scale_name])
            
            # Compute statistics for this scale's KL divergence across the epoch
            mean_kl = kl_values.mean()
            std_kl = kl_values.std()
            min_kl = kl_values.min()
            max_kl = kl_values.max()
            median_kl = np.median(kl_values)
            
            epoch_kl_stats[scale_name] = {
                'mean': mean_kl,
                'std': std_kl,
                'min': min_kl,
                'max': max_kl,
                'median': median_kl
            }
            
            writer.add_scalar(
                f'Epoch_KL/scale_{scale_name}_mean',
                mean_kl,
                global_step=epoch + 1
            )
            writer.add_scalar(
                f'Epoch_KL/scale_{scale_name}_std',
                std_kl,
                global_step=epoch + 1
            )
            writer.add_scalar(
                f'Epoch_KL/scale_{scale_name}_min',
                min_kl,
                global_step=epoch + 1
            )
            writer.add_scalar(
                f'Epoch_KL/scale_{scale_name}_max',
                max_kl,
                global_step=epoch + 1
            )
            writer.add_scalar(
                f'Epoch_KL/scale_{scale_name}_median',
                median_kl,
                global_step=epoch + 1
            )
        
        # Log all scales' mean KL on same plot for comparison
        writer.add_scalars(
            'Epoch_KL/all_scales_mean',
            {scale_name: epoch_kl_stats[scale_name]['mean'] for scale_name in scale_names},
            global_step=epoch + 1
        )
        
        # Log all scales' std on same plot
        writer.add_scalars(
            'Epoch_KL/all_scales_std',
            {scale_name: epoch_kl_stats[scale_name]['std'] for scale_name in scale_names},
            global_step=epoch + 1
        )
        
        # Compute inter-scale variance to detect if scales are converging to same KL
        mean_kls = np.array([epoch_kl_stats[s]['mean'] for s in scale_names])
        inter_scale_variance = mean_kls.var()
        inter_scale_std = mean_kls.std()
        
        writer.add_scalar(
            'Epoch_KL/inter_scale_variance',
            inter_scale_variance,
            global_step=epoch + 1
        )
        writer.add_scalar(
            'Epoch_KL/inter_scale_std',
            inter_scale_std,
            global_step=epoch + 1
        )
    if get_pseudolabel_acc:
        overall_acc = (pseudolabel_accuracy_tracker['correct'] / pseudolabel_accuracy_tracker['total']) * 100
        for scale_name in scale_names_list:
            correct = pseudolabel_accuracy_tracker['per_scale'][scale_name]['correct']
            total = pseudolabel_accuracy_tracker['per_scale'][scale_name]['total']
            acc = (correct / total) * 100 if total > 0 else 0
        writer.add_scalar(tag="Train/pseudolabel_overall_accuracy", scalar_value=overall_acc, global_step=epoch + 1)
        writer.add_scalars(main_tag="Train/pseudolabel_accuracy_all_scales",
                            tag_scalar_dict={scale_name: (pseudolabel_accuracy_tracker['per_scale'][scale_name]['correct'] / 
                                                        pseudolabel_accuracy_tracker['per_scale'][scale_name]['total']) * 100
                                            for scale_name in scale_names_list},
                            global_step=epoch + 1)
    return domain_weight


def test(target_domain, source_domains, test_dloader_list, model_list, epoch, writer, num_classes=126,
         top_5_accuracy=True, get_mmd=True):
    scales = ['final', 'scale1', 'scale2', 'scale4']
    
    source_domain_losses = [AverageMeter() for i in source_domains]
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.eval()
    # calculate loss, accuracy for target domain
    tmp_score = {k: [] for k in scales}
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    if get_mmd:
        scale_feats = ['feat_scale1', 'feat_scale2', 'feat_scale4']
        tmp_target_feats = {v: [] for v in scale_feats}
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            output_t = model_list[0](image_t)
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        task_loss_t = get_multiscale_classification_loss(output_t, label_t, task_criterion)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        for k in tmp_score.keys():
            tmp_score[k].append(torch.softmax(output_t[k], dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
        if get_mmd:
            for scale_feat in scale_feats:
                tmp_target_feats[scale_feat].append(output_t[scale_feat])
    if get_mmd:
        for scale_feat in scale_feats:
            tmp_target_feats[scale_feat] = torch.cat(tmp_target_feats[scale_feat], dim=0)
    writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
                      global_step=epoch + 1)
    # tmp_score = torch.cat([torch.cat(tmp_score[k], dim=0) for k in tmp_score.keys()], dim=1).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    for k in tmp_score.keys():
        tmp_score[k] = torch.cat(tmp_score[k], dim=0).detach()
        if top_5_accuracy:
            _, y_pred = torch.topk(tmp_score[k], k=5, dim=1)
        else:
            _, y_pred = torch.topk(tmp_score[k], k=1, dim=1)
        top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/target_domain_{}_{}_accuracy_top1".format(target_domain, k),
                        scalar_value=top_1_accuracy_t,
                        global_step=epoch + 1)
        if top_5_accuracy:
            top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(tag="Test/target_domain_{}_{}_accuracy_top5".format(target_domain, k),
                            scalar_value=top_5_accuracy_t,
                            global_step=epoch + 1)
            print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                            top_5_accuracy_t))
        else:
            print("Target Domain {} Scale {} Accuracy {:.3f}".format(target_domain, k, top_1_accuracy_t))


    # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):
        tmp_score = {k: [] for k in scales}
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        if get_mmd:
            tmp_source_feats = {v: [] for v in scale_feats}
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                output_s = model_list[s_i + 1](image_s)
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = get_multiscale_classification_loss(output_s, label_s, task_criterion)
            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
            for k in tmp_score.keys():
                tmp_score[k].append(torch.softmax(output_s[k], dim=1))
            # turn label into one-hot code
            tmp_label.append(label_onehot_s)
            if get_mmd:
                for scale_feat in scale_feats:
                    tmp_source_feats[scale_feat].append(output_s[scale_feat])
        writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)
        for k in tmp_score.keys():
            tmp_score[k] = torch.cat(tmp_score[k], dim=0).detach()
            if top_5_accuracy:
                _, y_pred = torch.topk(tmp_score[k], k=5, dim=1)
            else:
                _, y_pred = torch.topk(tmp_score[k], k=1, dim=1)
            top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
            writer.add_scalar(tag="Test/source_domain_{}_{}_accuracy_top1".format(domain_s, k), scalar_value=top_1_accuracy_s,
                            global_step=epoch + 1)
            if top_5_accuracy:
                top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
                writer.add_scalar(tag="Test/source_domain_{}_{}_accuracy_top5".format(domain_s, k), scalar_value=top_5_accuracy_s,
                                global_step=epoch + 1)
        # output MMD loss between target domain and each source domain
        if (get_mmd):
            for scale_feat in scale_feats:
                tmp_source_feats[scale_feat] = torch.cat(tmp_source_feats[scale_feat], dim=0)
                mmd = mmd_loss(tmp_source_feats[scale_feat], tmp_target_feats[scale_feat])
                writer.add_scalar(tag="Test/target_domain_{}_source_domain_{}_{}_mmd_loss".format(target_domain, domain_s, scale_feat),
                                scalar_value=mmd,
                                global_step=epoch + 1)