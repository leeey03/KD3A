import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter


def train(train_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, batch_per_epoch, top_5_accuracy=True):
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # Train model locally on source domains
    for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                    model_list[1:],
                                                                                    classifier_list[1:],
                                                                                    optimizer_list[1:],
                                                                                    classifier_optimizer_list[1:]):

        # for acc calculation 
        training_losses = AverageMeter()
        tmp_score = []
        tmp_label = []
        for i, (image_s, label_s) in enumerate(train_dloader):
            if i >= batch_per_epoch:
                break
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            # reset grad
            optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            # optimize
            feature_s = model(image_s)
            output_s = classifier(feature_s)
            task_loss_s = task_criterion(output_s, label_s)
            # get training acc and loss
            training_losses.update(float(task_loss_s.item()), image_s.size(0))
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            tmp_score.append(torch.softmax(output_s, dim=1))
            # turn label into one-hot code
            tmp_label.append(label_onehot_s)
            # backpropagation
            task_loss_s.backward()
            optimizer.step()
            classifier_optimizer.step()
        # TODO: fix bug on the source domain name for multiple source domains
        writer.add_scalar(tag="Train/source_domain_{}_loss".format(source_domains[0]), scalar_value=training_losses.avg,
                        global_step=epoch + 1)
        tmp_score = torch.cat(tmp_score, dim=0).detach()
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)
        if top_5_accuracy:
            _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        else:
            _, y_pred = torch.topk(tmp_score, k=1, dim=1)
        top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(tag="Train/source_domain_{}_accuracy_top1".format(source_domains[0]).format(source_domains[0]),
                        scalar_value=top_1_accuracy_s,
                        global_step=epoch + 1)
        if top_5_accuracy:
            top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(tag="Train/source_domain_{}_accuracy_top5".format(source_domains[0]).format(source_domains[0]),
                            scalar_value=top_5_accuracy_s,
                            global_step=epoch + 1)
            print("Source Domain Train {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(source_domains[0], top_1_accuracy_s,
                                                                            top_5_accuracy_s))
        else:
            print("Source Domain Train {} Accuracy {:.3f}".format(source_domains[0], top_1_accuracy_s))
    
    # Federated Averaging
    federated_average(model_list, domain_weight, batchnorm_mmd=False)



def test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes=126,
         top_5_accuracy=True, get_mmd=True):
    source_domain_losses = [AverageMeter() for i in source_domains]
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    if (get_mmd):
        features_t = []
        features_s = [[] for _ in source_domains]
    test_dloader_t = test_dloader_list[0]
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            feature_t = model_list[0](image_t)
            if (get_mmd):
                features_t.append(feature_t)
            output_t = classifier_list[0](feature_t)        
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        task_loss_t = task_criterion(output_t, label_t)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
    writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
                      global_step=epoch + 1)
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    if top_5_accuracy:
        _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    else:
        _, y_pred = torch.topk(tmp_score, k=1, dim=1)
    top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/target_domain_{}_accuracy_top1".format(target_domain).format(target_domain),
                      scalar_value=top_1_accuracy_t,
                      global_step=epoch + 1)
    if top_5_accuracy:
        top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/target_domain_{}_accuracy_top5".format(target_domain).format(target_domain),
                          scalar_value=top_5_accuracy_t,
                          global_step=epoch + 1)
        print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                          top_5_accuracy_t))
    else:
        print("Target Domain {} Accuracy {:.3f}".format(target_domain, top_1_accuracy_t))
    if (get_mmd):
        features_t = torch.cat(features_t, dim=0)

    # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):
        tmp_score = []
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                feature_s = model_list[s_i + 1](image_s)
                if (get_mmd):
                    features_s[s_i].append(feature_s)
                output_s = classifier_list[s_i + 1](feature_s)
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = task_criterion(output_s, label_s)
            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
            tmp_score.append(torch.softmax(output_s, dim=1))
            # turn label into one-hot code
            tmp_label.append(label_onehot_s)
        writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        tmp_score = torch.cat(tmp_score, dim=0).detach()
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)
        if top_5_accuracy:
            _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        else:
            _, y_pred = torch.topk(tmp_score, k=1, dim=1)
        top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/source_domain_{}_accuracy_top1".format(domain_s), scalar_value=top_1_accuracy_s,
                          global_step=epoch + 1)
        if top_5_accuracy:
            top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(tag="Test/source_domain_{}_accuracy_top5".format(domain_s), scalar_value=top_5_accuracy_s,
                              global_step=epoch + 1)
        if (get_mmd):
            features_s[s_i] = torch.cat(features_s[s_i], dim=0)
            mmd = mmd_loss(features_s[s_i], features_t)
            writer.add_scalar(tag="Test/target_domain_{}_source_domain_{}_mmd_loss".format(target_domain, domain_s),
                              scalar_value=mmd,
                              global_step=epoch + 1)