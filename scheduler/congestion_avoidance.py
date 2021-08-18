import torch 


def linear_cong_condition(min_cond, max_cond, epoch, max_epochs):

    condition = min_cond + (max_cond - min_cond) * (epoch / max_epochs)

    return condition


def congestion_avoid(model, optimizer, branch1_acc, branch2_acc, condition, branch_one_grads, branch_two_grads, min_epochs, mult):

    global epoch_count_one
    global epoch_count_two

    boolean_one = False
    boolean_two = False

    branch1_cond = (branch1_acc < condition * branch2_acc) and (epoch_count_two >= min_epochs)
    branch2_cond = (branch2_acc < condition * branch1_acc) and (epoch_count_one >= min_epochs)

    if branch1_cond:
        boolean_one = True
        print('Branch 1 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_two_grads.keys():
                    value += mult * branch_two_grads[name]
        epoch_count_two = 0
        #lr_two_cumulative = 0

    elif branch2_cond:
        boolean_two = True
        print('Branch 2 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_one_grads.keys():
                    value += mult * branch_one_grads[name]
        epoch_count_one = 0
        #lr_one_cumulative = 0
    
    else:
        print('No condition is met .....')

    return optimizer, model, boolean_one, boolean_two


def congestion_avoid_no_reset(model, optimizer, branch1_metric, branch2_metric, condition, branch_one_grads, branch_two_grads, min_epochs, mult):

    global epoch_count_one
    global epoch_count_two

    boolean_one = False
    boolean_two = False

    branch1_cond = (branch1_metric < condition * branch2_metric) and (epoch_count_two >= min_epochs)
    branch2_cond = (branch2_metric < condition * branch1_metric) and (epoch_count_one >= min_epochs)

    if branch1_cond:
        boolean_one = True
        print('Branch 1 condition has been met ..... {:.2f}%'.format(100.*condition))
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_two_grads.keys():
                    value += mult * branch_two_grads[name]
        for name in branch_two_grads.keys():
            branch_two_grads[name] -= mult * branch_two_grads[name]
        epoch_count_two = 0

    elif branch2_cond:
        boolean_two = True
        print('Branch 2 condition has been met ..... {:.2f}%'.format(100.*condition))
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_one_grads.keys():
                    value += mult * branch_one_grads[name]
        for name in branch_one_grads.keys():
            branch_one_grads[name] -= mult * branch_one_grads[name]
        epoch_count_one = 0
    
    else:
        print('No condition is met ..... {:.2f}%'.format(100.*condition))

    return optimizer, model, boolean_one, boolean_two, branch_one_grads, branch_two_grads


def congestion_avoid_weights(model, optimizer, branch1_acc, branch2_acc, condition, branch_one_weight_update, branch_two_weight_update, min_epochs, mult):

    global epoch_count_one
    global epoch_count_two

    boolean_one = False
    boolean_two = False

    branch1_cond = (branch1_acc < condition * branch2_acc) and (epoch_count_two >= min_epochs)
    branch2_cond = (branch2_acc < condition * branch1_acc) and (epoch_count_one >= min_epochs)

    if branch1_cond:
        boolean_one = True
        print('Branch 1 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_two_weight_update.keys():
                    value -= mult * branch_two_weight_update[name]

    elif branch2_cond:
        boolean_two = True
        print('Branch 2 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                if name in branch_one_weight_update.keys():
                    value -= mult * branch_one_weight_update[name]
    
    else:
        print('No condition is met .....')

    return optimizer, model, boolean_one, boolean_two
