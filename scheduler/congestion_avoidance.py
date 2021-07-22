import torch 


def congestion_avoid(model, optimizer, branch1_acc, branch2_acc, condition, branch_one_grads, branch_two_grads, min_epochs, mult):

    global epoch_count_one
    global epoch_count_two
    #global lr_one_cumulative
    #global lr_two_cumulative

    boolean_one = False
    boolean_two = False

    branch1_cond = (branch1_acc < condition * branch2_acc) and (epoch_count_two >= min_epochs)
    branch2_cond = (branch2_acc < condition * branch1_acc) and (epoch_count_one >= min_epochs)

    if branch1_cond:
        booelan_one = True
        print('Branch 1 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                value += mult * branch_two_grads[name]
        epoch_count_two = 0
        #lr_two_cumulative = 0

    elif branch2_cond:
        boolean_two = True
        print('Branch 2 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                value += mult * branch_one_grads[name]
        epoch_count_one = 0
        #lr_one_cumulative = 0
    
    else:
        print('No condition is met .....')

    return optimizer, model, boolean_one, boolean_two


def congestion_avoid_weights(model, optimizer, branch1_acc, branch2_acc, condition, branch_one_weight_updates, branch_two_weight_updates, roll_epochs):

    branch1_cond = branch1_acc < condition * branch2_acc
    branch2_cond = branch2_acc < condition * branch1_acc

    lr = optimizer.param_groups()[0]['lr']

    if branch1_cond:
        print('Branch 1 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                value -= roll_epochs * branch_two_weight_updates[name]

    elif branch2_cond:
        print('Branch 2 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                value -= roll_epochs * branch_one_weight_updates[name]
    
    else:
        print('No condition is met .....')

    return optimizer, model


def linear_cong_condition(min_cond, max_cond, epoch, max_epochs):

    condition = min_cond + (max_cond - min_cond) * (epoch / max_epochs)

    return condition


def congestion_avoid_weights_v2(model, optimizer, branch1_acc, branch2_acc, condition, branch_one_weight_update, branch_two_weight_update, min_epochs, mult):

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
                value -= mult * branch_two_weight_update[name]

    elif branch2_cond:
        boolean_two = True
        print('Branch 2 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                value -= mult * branch_one_weight_update[name]
    
    else:
        print('No condition is met .....')

    return optimizer, model, boolean_one, boolean_two