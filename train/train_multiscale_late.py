import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter


def train(train_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, 
          l2_kd_weight=0.1, mix_aug=True):
    """
    Multi-scale K3DA training with late fusion and L2 loss on KL divergences.
    
    Each domain has 3 models (scale1, scale4, scale16) that are trained with K3DA individually,
    plus a fusion classifier that combines all scales for final prediction.
    
    Args:
        model_list: List of dicts, one per domain: [{'scale1': model, 'scale4': model, 'scale16': model}, ...]
        classifier_list: List of dicts per domain: [{'scale1': cls, 'scale4': cls, 'scale16': cls, 'fusion': cls}, ...]
        optimizer_list: List of dicts per domain: [{'scale1': opt, 'scale4': opt, 'scale16': opt}, ...]
        classifier_optimizer_list: List of dicts per domain: [{'scale1': opt, ..., 'fusion': opt}, ...]
    """
    scale_names = ['scale1', 'scale4', 'scale16']
    task_criterion = nn.CrossEntropyLoss().cuda()
    # kl_criterion = nn.KLDivLoss(reduction='batchmean')
    source_domain_num = len(train_dloader_list[1:])
    
    # Set all models to train mode
    for model_dict in model_list:
        for scale in scale_names:
            model_dict[scale].train()
    for classifier_dict in classifier_list:
        for scale in scale_names:
            classifier_dict[scale].train()
        classifier_dict['fusion'].train()
    
    # Communication rounds handling
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    
    # Track KL divergence statistics for logging
    epoch_kl_stats = {scale: [] for scale in scale_names}
    epoch_l2_kd_loss = []
    
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        
        # Train models locally on source domains (per-scale K3DA)
        for train_dloader, model_dict, classifier_dict, optimizer_dict, classifier_optimizer_dict in zip(
                train_dloader_list[1:],
                model_list[1:],
                classifier_list[1:],
                optimizer_list[1:],
                classifier_optimizer_list[1:]):
            
            # Check for poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            poisoning_attack = (source_domain == malicious_domain and attack_level > 0)
            
            for i, (image_s, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break
                
                image_s = image_s.cuda()
                label_s = label_s.long().cuda()
                
                if poisoning_attack:
                    corrupted_num = round(label_s.size(0) * attack_level)
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                
                # Train each scale independently with standard classification loss
                scale_losses = []
                scale_features = []
                
                for scale in scale_names:
                    # Reset gradients
                    optimizer_dict[scale].zero_grad()
                    classifier_optimizer_dict[scale].zero_grad()
                    
                    # Forward pass
                    feature_s = model_dict[scale](image_s)
                    output_s = classifier_dict[scale](feature_s)
                    
                    # Classification loss
                    task_loss_s = task_criterion(output_s, label_s)
                    scale_losses.append(task_loss_s)
                    scale_features.append(feature_s)
                    
                    # Backward and optimize for this scale
                    task_loss_s.backward()
                    optimizer_dict[scale].step()
                    classifier_optimizer_dict[scale].step()
                
                # Train fusion classifier (late fusion)
                classifier_optimizer_dict['fusion'].zero_grad()
                
                # Concatenate features from all scales
                with torch.no_grad():
                    # Re-compute features (they're detached now)
                    scale_features = [model_dict[scale](image_s) for scale in scale_names]
                
                # fused_features = torch.cat(scale_features, dim=1)  # [B, D*3]
                fused_output = classifier_dict['fusion'](scale_features)
                fusion_loss = task_criterion(fused_output, label_s)
                fusion_loss.backward()
                classifier_optimizer_dict['fusion'].step()
    
    # Domain adaptation on target domain with multi-scale K3DA
    confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    target_weight = [0, 0]
    consensus_focus_dict = {}
    
    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0
    
    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        if i >= batch_per_epoch:
            break
        
        image_t = image_t.cuda()
        
        # Knowledge Vote: Perform SEPARATE knowledge voting for EACH scale
        # This follows K3DA's methodology where voting happens per prediction task
        with torch.no_grad():
            # For each scale, collect predictions from all source domains
            # B: batch size, C: num classes
            scale_knowledge_lists = {}  # {scale: [B, source_domain_num, C]}
            scale_consensus = {}  # {scale: consensus_knowledge [B, C]}
            scale_consensus_weights = {}  # {scale: consensus_weight [B]}
            
            for scale in scale_names:
                # Gather predictions from all source domains for this scale
                knowledge_list_for_scale = []
                for domain_idx in range(1, source_domain_num + 1):
                    feature = model_list[domain_idx][scale](image_t)
                    output = classifier_list[domain_idx][scale](feature)
                    pred = torch.softmax(output, dim=1).unsqueeze(1)  # [B, 1, C]
                    knowledge_list_for_scale.append(pred)
                
                # Concatenate: [B, source_domain_num, C]
                knowledge_list_for_scale = torch.cat(knowledge_list_for_scale, dim=1)
                scale_knowledge_lists[scale] = knowledge_list_for_scale
                
                # Perform K3DA knowledge voting for this scale
                _, consensus_knowledge, consensus_weight = knowledge_vote(
                    knowledge_list_for_scale, confidence_gate, num_classes=num_classes)
                
                scale_consensus[scale] = consensus_knowledge  # [B, C]
                scale_consensus_weights[scale] = consensus_weight  # [B]
        
        # Use the UNION of confident samples across all scales for target weight
        # A sample is confident if ANY scale has consensus
        combined_consensus_weight = torch.zeros_like(scale_consensus_weights['scale1'])
        for scale in scale_names:
            combined_consensus_weight = torch.maximum(combined_consensus_weight, 
                                                     scale_consensus_weights[scale])
        
        # TODO: check if this should be done per scale as well
        target_weight[0] += torch.sum(combined_consensus_weight).item()
        target_weight[1] += combined_consensus_weight.size(0)
        
        # Mixup augmentation
        if mix_aug:
            lam = np.random.beta(2, 2)
        else:
            lam = 1.0
        
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        
        # Apply mixup to each scale's consensus separately
        mixed_scale_consensus = {}
        for scale in scale_names:
            mixed_scale_consensus[scale] = (lam * scale_consensus[scale] + 
                                           (1 - lam) * scale_consensus[scale][index, :])  # [B, C]
        
        # Train each scale with its corresponding K3DA consensus pseudo-label
        # Step 1: Forward pass through all scales (keep gradients for L2 loss)
        scale_kl_losses = []
        scale_features = []
        scale_outputs = []
        
        for scale in scale_names:
            # Forward pass
            feature_t = model_list[0][scale](mixed_image)
            output_t = classifier_list[0][scale](feature_t)
            
            # KL divergence loss with K3DA consensus pseudo-label for this scale
            output_log_softmax = F.log_softmax(output_t, dim=1)
            pseudo_label = mixed_scale_consensus[scale]  # [B, C]
            
            # Per-sample KL divergence
            kl_per_sample = torch.sum(-1 * pseudo_label * output_log_softmax, dim=1)  # [B]
            kl_mean = kl_per_sample.mean()
            scale_kl_losses.append(kl_mean)  # Keep gradient connection!
            
            # Store for later use
            scale_features.append(feature_t)
            scale_outputs.append(output_t)
        
        # Step 2: Compute L2 loss on KL divergences (with gradients intact)
        kl_losses_tensor = torch.stack(scale_kl_losses)  # [3]
        mean_kl = kl_losses_tensor.mean()
        kl_variance = ((kl_losses_tensor - mean_kl) ** 2).mean()
        l2_kd_loss = torch.sqrt(kl_variance)
        
        # Step 3: Reset all gradients before joint optimization
        for scale in scale_names:
            optimizer_list[0][scale].zero_grad()
            classifier_optimizer_list[0][scale].zero_grad()
        
        # Step 4: Compute total loss with L2 regularization for joint optimization
        # This ensures L2 loss gradients flow back to all scale parameters
        total_scale_loss = 0
        
        for scale_idx, scale in enumerate(scale_names):
            # Recompute KL loss for gradient flow (using stored outputs)
            output_log_softmax = F.log_softmax(scale_outputs[scale_idx], dim=1)
            pseudo_label = mixed_scale_consensus[scale]
            kl_per_sample = torch.sum(-1 * pseudo_label * output_log_softmax, dim=1)
            
            # Weight by confidence from THIS scale's knowledge vote
            weighted_kl = torch.mean(scale_consensus_weights[scale] * kl_per_sample)
            
            # Add L2 loss to each scale (KEY: this connects all scales through L2)
            # When we backprop through l2_kd_loss, gradients flow to all scales
            scale_loss = weighted_kl + l2_kd_weight * l2_kd_loss
            total_scale_loss += scale_loss
        
        # Step 5: Single backward pass - gradients flow to all scales including L2 term
        # ALT: implement per-scale backward passes (but less coordination between the scales if done this way?)
        total_scale_loss.backward()
        
        # Step 6: Update all scale models
        for scale in scale_names:
            optimizer_list[0][scale].step()
            classifier_optimizer_list[0][scale].step()
        
        # Train fusion classifier with ensemble pseudo-label
        # Fusion pseudo-label strategy: Confidence-weighted averaging across scales
        classifier_optimizer_list[0]['fusion'].zero_grad()
        
        # Concatenate features from all scales (detach since we already updated encoders)
        with torch.no_grad():
            scale_features_detached = [model_list[0][scale](mixed_image) for scale in scale_names]
        
        # fused_features = torch.cat(scale_features_detached, dim=1)  # [B, D*3]
        fused_output = classifier_list[0]['fusion'](scale_features_detached)
        fused_output_log_softmax = F.log_softmax(fused_output, dim=1)
        
        # Create ensemble pseudo-label via confidence-weighted averaging
        # This gives more weight to scales with higher consensus
        total_weight = 0
        weighted_pseudo_sum = torch.zeros_like(mixed_scale_consensus['scale1'])
        
        for scale in scale_names:
            # Weight each scale's pseudo-label by its consensus weight
            scale_weight = scale_consensus_weights[scale].unsqueeze(1)  # [B, 1]
            weighted_pseudo = scale_weight * mixed_scale_consensus[scale]  # [B, C]
            weighted_pseudo_sum += weighted_pseudo
            total_weight += scale_weight
        
        # Normalize to get ensemble pseudo-label
        ensemble_pseudo_label = weighted_pseudo_sum / (total_weight + 1e-8)  # [B, C]
        
        # Use combined consensus weight (union of confident samples)
        fusion_consensus_weight = combined_consensus_weight
        
        # KL divergence for fusion
        kl_per_sample_fusion = torch.sum(-1 * ensemble_pseudo_label * fused_output_log_softmax, dim=1)
        fusion_loss = torch.mean(fusion_consensus_weight * kl_per_sample_fusion)
        
        # Backward and optimize fusion (no L2 loss needed here since scales already optimized)
        fusion_loss.backward()
        classifier_optimizer_list[0]['fusion'].step()
        
        # Track statistics
        for scale_idx, scale in enumerate(scale_names):
            epoch_kl_stats[scale].append(scale_kl_losses[scale_idx].item())
        epoch_l2_kd_loss.append(l2_kd_loss.item())
        
        # Calculate consensus focus (aggregate across all scales)
        # We need to convert scale_knowledge_lists back to the format expected by calculate_consensus_focus
        # Aggregate predictions per domain (average across scales)
        # ALT: Implement multiscale consensus focus calculation instead
        aggregated_knowledge_list = []
        for domain_idx in range(source_domain_num):
            # Collect this domain's predictions across all scales
            domain_preds = []
            for scale in scale_names:
                domain_scale_pred = scale_knowledge_lists[scale][:, domain_idx, :]  # [B, C]
                domain_preds.append(domain_scale_pred)
            
            # Average across scales for this domain
            domain_avg_pred = torch.stack(domain_preds, dim=0).mean(dim=0)  # [B, C]
            aggregated_knowledge_list.append(domain_avg_pred.unsqueeze(1))  # [B, 1, C]

        aggregated_knowledge_list = torch.cat(aggregated_knowledge_list, dim=1)  # [B, source_domain_num, C]
        
        consensus_focus_dict = calculate_consensus_focus(
            consensus_focus_dict, aggregated_knowledge_list, confidence_gate,
            source_domain_num, num_classes)
    
    # Consensus Focus Re-weighting
    target_parameter_alpha = target_weight[0] / target_weight[1]
    target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    epoch_domain_weight = []
    source_total_weight = 1 - target_weight
    
    for i in range(1, source_domain_num + 1):
        epoch_domain_weight.append(consensus_focus_dict[i])
    
    if sum(epoch_domain_weight) == 0:
        epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
    
    epoch_domain_weight = [
        round(source_total_weight * v / sum(epoch_domain_weight), 4)
        for v in epoch_domain_weight
    ]
    epoch_domain_weight.insert(0, target_weight)
    
    # Update domain weight with moving average
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)
    
    # Model aggregation per scale (federated averaging)
    for scale in scale_names:
        scale_models = [model_dict[scale] for model_dict in model_list]
        federated_average(scale_models, domain_weight, batchnorm_mmd=batchnorm_mmd)
        
        # Also aggregate classifiers per scale
        scale_classifiers = [classifier_dict[scale] for classifier_dict in classifier_list]
        federated_average(scale_classifiers, domain_weight, batchnorm_mmd=batchnorm_mmd)
    
    # Aggregate fusion classifiers
    fusion_classifiers = [classifier_dict['fusion'] for classifier_dict in classifier_list]
    federated_average(fusion_classifiers, domain_weight, batchnorm_mmd=batchnorm_mmd)
    
    # Logging
    writer.add_scalar("Train/target_domain_weight", target_weight, epoch + 1)
    for i in range(len(source_domains)):
        writer.add_scalar(f"Train/source_domain_{source_domains[i]}_weight",
                         domain_weight[i + 1], epoch + 1)
    
    # Log KL divergence statistics
    for scale in scale_names:
        kl_values = np.array(epoch_kl_stats[scale])
        if len(kl_values) > 0:
            writer.add_scalar(f'Train_KD/scale_{scale}_mean_kl', kl_values.mean(), epoch + 1)
            writer.add_scalar(f'Train_KD/scale_{scale}_std_kl', kl_values.std(), epoch + 1)
    
    # Log L2 KD loss
    if len(epoch_l2_kd_loss) > 0:
        l2_kd_values = np.array(epoch_l2_kd_loss)
        writer.add_scalar('Train_KD/l2_kd_loss_mean', l2_kd_values.mean(), epoch + 1)
        writer.add_scalar('Train_KD/l2_kd_loss_std', l2_kd_values.std(), epoch + 1)
        
        # Log inter-scale KL variance
        mean_kls = np.array([np.array(epoch_kl_stats[s]).mean() for s in scale_names if len(epoch_kl_stats[s]) > 0])
        if len(mean_kls) > 0:
            inter_scale_variance = mean_kls.var()
            writer.add_scalar('Train_KD/inter_scale_kl_variance', inter_scale_variance, epoch + 1)
            
            print(f"Source Domains: {source_domains}, Domain Weight: {domain_weight[1:]}")
            print(f"Scale KL means: {[f'{k:.4f}' for k in mean_kls]}")
            print(f"Inter-scale KL variance: {inter_scale_variance:.4f}")
            print(f"L2 KD loss mean: {l2_kd_values.mean():.4f}")
    
    return domain_weight


def test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, 
         num_classes=126, top_5_accuracy=True, get_mmd=True):
    """
    Test function for multi-scale K3DA with late fusion.
    
    Args:
        model_list: List of dicts per domain: [{'scale1': model, 'scale4': model, 'scale16': model}, ...]
        classifier_list: List of dicts per domain: [{'scale1': cls, ..., 'fusion': cls}, ...]
    """
    scale_names = ['scale1', 'scale4', 'scale16']
    
    source_domain_losses = {
        scale: [AverageMeter() for _ in source_domains] 
        for scale in scale_names + ['fusion']
    }
    target_domain_losses = {scale: AverageMeter() for scale in scale_names + ['fusion']}
    task_criterion = nn.CrossEntropyLoss().cuda()
    
    # Set all models to eval mode
    for model_dict in model_list:
        for scale in scale_names:
            model_dict[scale].eval()
    for classifier_dict in classifier_list:
        for scale in scale_names:
            classifier_dict[scale].eval()
        classifier_dict['fusion'].eval()
    
    # Test on target domain
    tmp_score = {scale: [] for scale in scale_names + ['fusion']}
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    
    if get_mmd:
        features_t = {scale: [] for scale in scale_names}
    
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        
        with torch.no_grad():
            scale_features = []
            
            # Process each scale
            for scale in scale_names:
                feature_t = model_list[0][scale](image_t)
                output_t = classifier_list[0][scale](feature_t)
                
                # Store for evaluation
                tmp_score[scale].append(torch.softmax(output_t, dim=1))
                scale_features.append(feature_t)
                
                # Compute loss
                task_loss = task_criterion(output_t, label_t)
                target_domain_losses[scale].update(task_loss.item(), image_t.size(0))
                
                if get_mmd:
                    features_t[scale].append(feature_t)
            
            # Fusion prediction
            # fused_features = torch.cat(scale_features, dim=1)
            fused_output = classifier_list[0]['fusion'](scale_features)
            tmp_score['fusion'].append(torch.softmax(fused_output, dim=1))
            
            fusion_loss = task_criterion(fused_output, label_t)
            target_domain_losses['fusion'].update(fusion_loss.item(), image_t.size(0))
        
        # Label (only need once)
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(
            1, label_t.view(-1, 1), 1)
        tmp_label.append(label_onehot_t)
    
    # Log target domain results
    for scale in scale_names + ['fusion']:
        writer.add_scalar(f"Test/target_domain_{target_domain}_{scale}_loss",
                         target_domain_losses[scale].avg, epoch + 1)
    
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    
    # Compute accuracy per scale and fusion
    for scale in scale_names + ['fusion']:
        tmp_score[scale] = torch.cat(tmp_score[scale], dim=0).detach()
        
        if top_5_accuracy:
            _, y_pred = torch.topk(tmp_score[scale], k=5, dim=1)
        else:
            _, y_pred = torch.topk(tmp_score[scale], k=1, dim=1)
        
        top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(f"Test/target_domain_{target_domain}_{scale}_accuracy_top1",
                         top_1_accuracy_t, epoch + 1)
        
        if top_5_accuracy:
            top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(f"Test/target_domain_{target_domain}_{scale}_accuracy_top5",
                             top_5_accuracy_t, epoch + 1)
            print(f"Target Domain {target_domain} {scale} - "
                  f"Top1: {top_1_accuracy_t:.3f} Top5: {top_5_accuracy_t:.3f}")
        else:
            print(f"Target Domain {target_domain} {scale} - "
                  f"Accuracy: {top_1_accuracy_t:.3f}")
    
    if get_mmd:
        for scale in scale_names:
            features_t[scale] = torch.cat(features_t[scale], dim=0)
    
    # Test on source domains
    for s_i, domain_s in enumerate(source_domains):
        tmp_score = {scale: [] for scale in scale_names + ['fusion']}
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        
        if get_mmd:
            features_s = {scale: [] for scale in scale_names}
        
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            
            with torch.no_grad():
                scale_features = []
                
                for scale in scale_names:
                    feature_s = model_list[s_i + 1][scale](image_s)
                    output_s = classifier_list[s_i + 1][scale](feature_s)
                    
                    tmp_score[scale].append(torch.softmax(output_s, dim=1))
                    scale_features.append(feature_s)
                    
                    task_loss = task_criterion(output_s, label_s)
                    source_domain_losses[scale][s_i].update(task_loss.item(), image_s.size(0))
                    
                    if get_mmd:
                        features_s[scale].append(feature_s)
                
                # Fusion
                # fused_features = torch.cat(scale_features, dim=1)
                fused_output = classifier_list[s_i + 1]['fusion'](scale_features)
                tmp_score['fusion'].append(torch.softmax(fused_output, dim=1))
                
                fusion_loss = task_criterion(fused_output, label_s)
                source_domain_losses['fusion'][s_i].update(fusion_loss.item(), image_s.size(0))
            
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(
                1, label_s.view(-1, 1), 1)
            tmp_label.append(label_onehot_s)
        
        # Log source domain results
        for scale in scale_names + ['fusion']:
            writer.add_scalar(f"Test/source_domain_{domain_s}_{scale}_loss",
                             source_domain_losses[scale][s_i].avg, epoch + 1)
        
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)
        
        for scale in scale_names + ['fusion']:
            tmp_score[scale] = torch.cat(tmp_score[scale], dim=0).detach()
            
            if top_5_accuracy:
                _, y_pred = torch.topk(tmp_score[scale], k=5, dim=1)
            else:
                _, y_pred = torch.topk(tmp_score[scale], k=1, dim=1)
            
            top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
            writer.add_scalar(f"Test/source_domain_{domain_s}_{scale}_accuracy_top1",
                             top_1_accuracy_s, epoch + 1)
            
            if top_5_accuracy:
                top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
                writer.add_scalar(f"Test/source_domain_{domain_s}_{scale}_accuracy_top5",
                                 top_5_accuracy_s, epoch + 1)
        
        # MMD computation per scale
        if get_mmd:
            for scale in scale_names:
                features_s[scale] = torch.cat(features_s[scale], dim=0)
                mmd = mmd_loss(features_s[scale], features_t[scale])
                writer.add_scalar(
                    f"Test/target_domain_{target_domain}_source_domain_{domain_s}_{scale}_mmd_loss",
                    mmd, epoch + 1)