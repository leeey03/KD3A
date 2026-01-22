import math
import random

random.seed(1)
import numpy as np

np.random.seed(1)

import argparse
from lib.utils.federated_utils import *
from train.train_multiscale import train, test
# from model.epickitchens import EpicKitchensTransformerEncoder, EpicKitchensTransformerClassifier
from model.multiscale import MultiScaleTemporalTransformer
from datasets.EpicKitchens import get_epic_dloader
import os
from os import path
import shutil
import yaml
import time

# Only use F16 tensors like local GPU
torch.backends.cuda.matmul.allow_tf32 = False      # full fp32
torch.backends.cudnn.allow_tf32 = False
# deteministic mode
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Default settings
parser = argparse.ArgumentParser(description='Multi-Scale KD3A')
# Dataset Parameters
parser.add_argument("--config", default="DigitFive.yaml")
parser.add_argument('-bp', '--base-path', default="./") 
parser.add_argument('--target-domain', type=str, help="The target domain we want to perform domain adaptation")
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=str,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
parser.add_argument('-mmd', '--get-mmd', action='store_true', help='Get MMD Loss')
parser.add_argument('-kl', '--get-kl', action='store_true', help='Get KL Loss values per scale')
parser.add_argument('-pl', '--get-pseudolabel-acc', action='store_true', help='Get Pseudolabel Accuracy')

args = parser.parse_args()
# import config files
with open(r"./config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args=args, configs=configs):
    # set the dataloader list, model list, optimizer list, optimizer schedule list
    train_dloaders = []
    test_dloaders = []
    models = []
    optimizers = []
    optimizer_schedulers = []
    # build dataset
    if configs["DataConfig"]["dataset"] == "EpicKitchens":
        domains = args.source_domains  # source domains
        # [0]: target dataset, target backbone, [1:-1]: source dataset, source backbone
        target_train_dloader, target_test_dloader = get_epic_dloader(
            train_list="data/frame_annotations_transVAE/list_{}_train.txt".format(args.target_domain), # should be P22
            test_list="data/frame_annotations_transVAE/list_{}_test.txt".format(args.target_domain),
            batch_size=configs["TrainingConfig"]["batch_size"],
            num_segments=configs["DataConfig"]["num_segments"],
            num_workers=args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        model = MultiScaleTemporalTransformer(d_model=512, n_heads=8, n_layers=4, dim_feedforward=1024, classes=8, dropout=0.1, input_dim=2048).cuda()
        if args.data_parallel:
            device_ids = list(range(torch.cuda.device_count()))
            if len(device_ids) < 2:
                print(f"Warning: Only {len(device_ids)} GPU(s) available. Data parallelism may not be beneficial.")
            print(f"Using DataParallel on GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        models.append(model)
        
        for domain in domains:
            source_train_dloader, source_test_dloader = get_epic_dloader(
                train_list="data/frame_annotations_transVAE/list_{}_train.txt".format(domain),
                test_list="data/frame_annotations_transVAE/list_{}_test.txt".format(domain),
                data_dir=configs["DataConfig"]["data_dir"],
                batch_size=configs["TrainingConfig"]["batch_size"],
                num_segments=configs["DataConfig"]["num_segments"],
                num_workers=args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            model = MultiScaleTemporalTransformer(d_model=512, n_heads=8, n_layers=4, dim_feedforward=1024, classes=8, dropout=0.1, input_dim=2048).cuda()
            if args.data_parallel:
                device_ids = list(range(torch.cuda.device_count()))
                if len(device_ids) < 2:
                    print(f"Warning: Only {len(device_ids)} GPU(s) available. Data parallelism may not be beneficial.")
                print(f"Using DataParallel on GPUs: {device_ids}")
                model = nn.DataParallel(model, device_ids=device_ids)
            models.append(model)
        num_classes = 8
    else:
        raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))
    # federated learning step 1: initialize model with the same parameter (use target as standard)
    for model in models[1:]:
        for source_weight, target_weight in zip(model.named_parameters(), models[0].named_parameters()):
            # consistent parameters
            source_weight[1].data = target_weight[1].data.clone()
    # create the optimizer for each model
    for model in models:
        # check if model is using GPU
        print(f"Model {model.module.name if isinstance(model, nn.DataParallel) else model.name} device:", next(model.parameters()).device)
        optimizers.append(
            torch.optim.SGD(model.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    # for classifier in classifiers:
    #     print(f"Classifier {classifier.name} device:", next(classifier.parameters()).device)
    #     classifier_optimizers.append(
    #         torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
    #                         lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    # create the optimizer scheduler with cosine annealing schedule
    for optimizer in optimizers:
        optimizer_schedulers.append(
            CosineAnnealingLR(optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))
    # for classifier_optimizer in classifier_optimizers:
    #     classifier_optimizer_schedulers.append(
    #         CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
    #                           eta_min=configs["TrainingConfig"]["learning_rate_end"]))
    # create the event to save log info
    writer_log_dir = os.path.join(args.base_path, configs["DataConfig"]["dataset"], "runs",
                               "train_time_{}".format(args.train_time) + "_" +
                               args.target_domain + "_" + "_".join(args.source_domains))
    print("create writer in {}".format(writer_log_dir))
    if os.path.exists(writer_log_dir):
        flag = input("{} train_time:{} will be removed, input yes to continue:".format(
            configs["DataConfig"]["dataset"], args.train_time))
        if flag == "yes":
            shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    
    # check for GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}\n")   # Should be True
    print(f"GPU Device Count: {torch.cuda.device_count()}\n")   # Should be >= 1
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}\n")  # Should print e.g. "A100" or "V100"

    # begin train
    print("Begin the {} time's training, Dataset:{}, Source Domains {}, Target Domain {}".format(args.train_time,
                                                                                                 configs[
                                                                                                     "DataConfig"][
                                                                                                     "dataset"],
                                                                                                 args.source_domains,
                                                                                                 args.target_domain))

    # create the initialized domain weight
    domain_weight = create_domain_weight(len(args.source_domains))
    print("Initial domain weight {}".format(domain_weight))
    # adjust training strategy with communication round
    batch_per_epoch, total_epochs = decentralized_training_strategy(
        communication_rounds=configs["UMDAConfig"]["communication_rounds"],
        epoch_samples=configs["TrainingConfig"]["epoch_samples"],
        batch_size=configs["TrainingConfig"]["batch_size"],
        total_epochs=configs["TrainingConfig"]["total_epochs"])
    # train model
    train_start_time = time.time()
    for epoch in range(args.start_epoch, total_epochs):
        torch.cuda.reset_peak_memory_stats()
        # epoch_start_mem = torch.cuda.memory_allocated() / 1024**2
        epoch_start_time = time.time()
        # include pytoch profiler
        domain_weight = train(train_dloaders, models, optimizers,
                            epoch, writer, num_classes=num_classes,
                            domain_weight=domain_weight, source_domains=args.source_domains,
                            batch_per_epoch=batch_per_epoch, total_epochs=total_epochs,
                            batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
                            communication_rounds=configs["UMDAConfig"]["communication_rounds"],
                            confidence_gate_begin=configs["UMDAConfig"]["confidence_gate_begin"],
                            confidence_gate_end=configs["UMDAConfig"]["confidence_gate_end"],
                            malicious_domain=configs["UMDAConfig"]["malicious"]["attack_domain"],
                            attack_level=configs["UMDAConfig"]["malicious"]["attack_level"],
                            tau=configs["ModelConfig"]["tau"],
                            mix_aug=(configs["DataConfig"]["dataset"] != "AmazonReview"),
                            get_KL_values=args.get_kl,
                            get_pseudolabel_acc=args.get_pseudolabel_acc)
        test(args.target_domain, args.source_domains, test_dloaders, models, epoch,
            writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10), get_mmd=args.get_mmd)
        for scheduler in optimizer_schedulers:
            scheduler.step()
        # save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            # save target model with epoch, domain, model, optimizer
            save_checkpoint(
                {"epoch": epoch + 1,
                "domain": args.target_domain,
                "backbone": models[0].state_dict(),
                "optimizer": optimizers[0].state_dict(),
                },
                filename="{}.pth.tar".format(args.target_domain))
        # peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        # end_mem = torch.cuda.memory_allocated() / 1024**2
        # print(f"[GPU Memory] Epoch {epoch}: start={epoch_start_mem:.2f}MB, "f"end={end_mem:.2f}MB, peak={peak_mem:.2f}MB")
        print(f"Training epoch {epoch} completed in {time.time() - epoch_start_time:.2f} seconds")
    print("Total training time is {:.2f} hours".format((time.time() - train_start_time) / 3600))

def save_checkpoint(state, filename):
    filefolder = "{}/{}/parameter/train_time:{}".format(args.base_path, configs["DataConfig"]["dataset"],
                                                        args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
