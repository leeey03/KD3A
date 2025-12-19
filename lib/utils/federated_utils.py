from itertools import permutations, combinations
import torch


def create_domain_weight(source_domain_num):
    global_federated_matrix = [1 / (source_domain_num + 1)] * (source_domain_num + 1)
    return global_federated_matrix


def update_domain_weight(global_domain_weight, epoch_domain_weight, momentum=0.9):
    global_domain_weight = [round(global_domain_weight[i] * momentum + epoch_domain_weight[i] * (1 - momentum), 4)
                            for i in range(len(epoch_domain_weight))]
    return global_domain_weight


def federated_average(model_list, coefficient_matrix, batchnorm_mmd=True):
    """
    :param model_list: a list of all models needed in federated average. [0]: model for target domain,
    [1:-1] model for source domains
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :param batchnorm_mmd: bool, if true, we use the batchnorm mmd
    :return model list after federated average
    """
    if batchnorm_mmd:
        dict_list = [it.state_dict() for it in model_list]
        dict_item_list = [dic.items() for dic in dict_list]
        for key_data_pair_list in zip(*dict_item_list):
            source_data_list = [pair[1] * coefficient_matrix[idx] for idx, pair in
                                enumerate(key_data_pair_list)]
            dict_list[0][key_data_pair_list[0][0]] = sum(source_data_list)
        for model in model_list:
            model.load_state_dict(dict_list[0])
    else:
        named_parameter_list = [model.named_parameters() for model in model_list]
        for parameter_list in zip(*named_parameter_list):
            source_parameters = [parameter[1].data.clone() * coefficient_matrix[idx] for idx, parameter in
                                 enumerate(parameter_list)]
            parameter_list[0][1].data = sum(source_parameters)
            for parameter in parameter_list[1:]:
                parameter[1].data = parameter_list[0][1].data.clone()


def knowledge_vote(knowledge_list, confidence_gate, num_classes):
    """
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :return: consensus_confidence,consensus_knowledge,consensus_knowledge_weight
    """
    max_p, max_p_class = knowledge_list.max(2)
    max_conf, _ = max_p.max(1)
    max_p_mask = (max_p > confidence_gate).float().cuda()
    consensus_knowledge = torch.zeros(knowledge_list.size(0), knowledge_list.size(2)).cuda()
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        # to solve the [0,0,0] situation
        if torch.sum(p_mask) > 0:
            p = p * p_mask
        for source_idx, source_class in enumerate(p_class):
            consensus_knowledge[batch_idx, source_class] += p[source_idx]
    consensus_knowledge_conf, consensus_knowledge = consensus_knowledge.max(1)
    consensus_knowledge_mask = (max_conf > confidence_gate).float().cuda()
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).cuda().scatter_(1,
                                                                                                consensus_knowledge.view(
                                                                                                    -1, 1), 1)
    return consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask

def knowledge_vote_multiscale(knowledge_list, confidence_gate, num_classes):
    """
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :return: consensus_confidence,consensus_knowledge,consensus_knowledge_weight
    """
    max_p, max_p_class = knowledge_list.max(2)
    max_conf, _ = max_p.max(1)
    max_p_mask = (max_p > confidence_gate).float().cuda()
    consensus_knowledge = torch.zeros(knowledge_list.size(0), knowledge_list.size(2)).cuda()
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        # to solve the [0,0,0] situation
        if torch.sum(p_mask) > 0:
            p = p * p_mask
        for source_idx, source_class in enumerate(p_class):
            consensus_knowledge[batch_idx, source_class] += p[source_idx]
    consensus_knowledge_conf, consensus_knowledge = consensus_knowledge.max(1)
    # consensus_knowledge_mask returns batch samples that pass confidence gate
    consensus_knowledge_mask = (max_conf > confidence_gate).float().cuda()
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).cuda().scatter_(1,
                                                                                                consensus_knowledge.view(
                                                                                                    -1, 1), 1)
    return consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask, max_p_mask

def calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate, source_domain_numbers,
                              num_classes):
    """
    :param consensus_focus_dict: record consensus_focus for each domain
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :param source_domain_numbers: the numbers of source domains
    """
    domain_contribution = {frozenset(): 0}
    for combination_num in range(1, source_domain_numbers + 1):
        combination_list = list(combinations(range(source_domain_numbers), combination_num))
        for combination in combination_list:
            consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask = knowledge_vote(
                knowledge_list[:, combination, :], confidence_gate, num_classes)
            domain_contribution[frozenset(combination)] = torch.sum(
                consensus_knowledge_conf * consensus_knowledge_mask).item()
    permutation_list = list(permutations(range(source_domain_numbers), source_domain_numbers))
    permutation_num = len(permutation_list)
    for permutation in permutation_list:
        permutation = list(permutation)
        for source_idx in range(source_domain_numbers):
            consensus_focus_dict[source_idx + 1] += (domain_contribution[frozenset(permutation[:permutation.index(source_idx) + 1])]
                                                    - domain_contribution[frozenset(permutation[:permutation.index(source_idx)])]
                                                    ) / permutation_num
    return consensus_focus_dict

def calculate_multiscale_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate, source_domain_numbers, num_scales=4):
    """
    modified calculate_consensus_focus to handle multi-scale knowledge. 
    Performs per scale knowledge vote then aggregates per source. Returns consensus_focus_dict with same struture as original
    """
    B, N, C = knowledge_list.shape
    domain_contribution = {frozenset(): 0.0}
    for combination_num in range(1, source_domain_numbers + 1):
        for combination in combinations(range(source_domain_numbers), combination_num):
            # get all scale indices for selected combination
            flat_indices = [] 
            for src in combination:
                flat_indices.extend([src * num_scales + s for s in range(num_scales)])
            # knowledge vote over all scales within selected combination
            consensus_conf, _, consensus_mask = knowledge_vote(knowledge_list[:, flat_indices, :], confidence_gate, C)
            # aggregate scale contributions per source 
            mask = consensus_mask.view(B, len(combination), num_scales)  
            conf = consensus_conf.unsqueeze(-1).expand_as(mask)
            contribution = torch.sum(conf * mask).item()
            domain_contribution[frozenset(combination)] = contribution
    permutation_list = list(permutations(range(source_domain_numbers), source_domain_numbers))
    permutation_num = len(permutation_list)
    for permutation in permutation_list:
        permutation = list(permutation)
        for source_idx in range(source_domain_numbers):
            consensus_focus_dict[source_idx + 1] += (domain_contribution[frozenset(permutation[:permutation.index(source_idx) + 1])]
                                                    - domain_contribution[frozenset(permutation[:permutation.index(source_idx)])]
                                                    ) / permutation_num
    return consensus_focus_dict

def calculate_temporal_consistency(knowledge_list, source_domain_num, num_scales=4):
    """
    Calculate temporal (cross-scale) consistency per source.
    """

    B, _, C = knowledge_list.shape
    temporal_consistency_dict = {i + 1: 0.0 for i in range(source_domain_num)}

    # Convert probabilities to hard predictions
    preds = knowledge_list.argmax(dim=-1)  # [B, M*S]

    for src_idx in range(source_domain_num):
        start = src_idx * num_scales
        end = start + num_scales

        # [B, S]
        src_preds = preds[:, start:end]

        # Pairwise agreement across scales
        for s1, s2 in combinations(range(num_scales), 2):
            agreement = (src_preds[:, s1] == src_preds[:, s2]).float()
            temporal_consistency_dict[src_idx + 1] += agreement.sum().item()

    return temporal_consistency_dict


def decentralized_training_strategy(communication_rounds, epoch_samples, batch_size, total_epochs):
    """
    Split one epoch into r rounds and perform model aggregation
    :param communication_rounds: the communication rounds in training process
    :param epoch_samples: the samples for each epoch
    :param batch_size: the batch_size for each epoch
    :param total_epochs: the total epochs for training
    :return: batch_per_epoch, total_epochs with communication rounds r
    """
    if communication_rounds >= 1:
        epoch_samples = round(epoch_samples / communication_rounds)
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    elif communication_rounds in [0.2, 0.5]:
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    else:
        raise NotImplementedError(
            "The communication round {} illegal, should be 0.2 or 0.5".format(communication_rounds))
    return batch_per_epoch, total_epochs

## below functions are generated with AI
def gaussian_kernel(x, y, var=1.0):
    # Calculate the RBF kernel matrix
    # Based on the formula: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    # where sigma^2 = var
    
    # Expand dimensions for pairwise difference
    x_expand = x.unsqueeze(1) # [B, 1, D]
    y_expand = y.unsqueeze(0) # [1, B, D]
    
    # Calculate pairwise squared distances
    distances = torch.sum((x_expand - y_expand)**2, dim=2) # [B, B]
    
    # Apply kernel function
    kernel = torch.exp(-distances / (2 * var))
    return kernel

def mmd_loss(x, y, var=1.0):
    """
    Compute the unbiased MMD^2 loss between two sets of samples x and y.
    Supports different batch sizes.
    
    Args:
        x: Tensor of shape [m, d]
        y: Tensor of shape [n, d]
        var: Kernel variance (sigma^2)
    Returns:
        Scalar tensor representing MMD^2 value.
    """
    m = x.size(0)
    n = y.size(0)

    # Compute kernel matrices
    k_xx = gaussian_kernel(x, x, var)
    k_yy = gaussian_kernel(y, y, var)
    k_xy = gaussian_kernel(x, y, var)

    # Remove diagonal for unbiased estimate (self-similarity)
    # Only valid if m>1 and n>1
    if m > 1:
        k_xx = k_xx - torch.diag(torch.diag(k_xx))
    if n > 1:
        k_yy = k_yy - torch.diag(torch.diag(k_yy))

    # Compute unbiased MMD^2 with different sample sizes
    term_xx = k_xx.sum() / (m * (m - 1)) if m > 1 else 0
    term_yy = k_yy.sum() / (n * (n - 1)) if n > 1 else 0
    term_xy = k_xy.sum() / (m * n)
    
    mmd2 = term_xx + term_yy - 2 * term_xy
    return mmd2

def get_multiscale_classification_loss(output, label, criterion):
    """
    Return average classification loss across all scales
    """
    return (criterion(output['final'], label) + criterion(output['scale1'], label) 
            + criterion(output['scale2'], label) + criterion(output['scale4'], label)) / 4