def congestion_control_scheduler(decay_factor, mult_factor, branch1_optim, branch2_optim, branch1_acc, branch2_acc, condition):
    '''
    A function to take the current optimizers for the ResNet model branches.
    If one branch is converging too slowly then the LR is multiplied by a factor,
    whilst the other branch is paused!!!!

    If no condition is met then the two LR's continue to exponentially decay.

    '''

    import math

    lr_1 = branch1_optim.param_groups[0]['lr']
    lr_2 = branch2_optim.param_groups[0]['lr']

    branch1_cond = branch1_acc < condition * branch2_acc
    branch2_cond = branch2_acc < condition * branch1_acc

    if branch1_cond:
        for param_group in branch1_optim.param_groups:
            param_group['lr'] = param_group['lr'] * mult_factor
        for param_group in branch2_optim.param_groups:
            param_group['lr'] = 0

    elif branch2_cond:
        for param_group in branch1_optim.param_groups:
            param_group['lr'] = 0
        for param_group in branch2_optim.param_groups:
            param_group['lr'] = param_group['lr'] * mult_factor
    
    else:
        for param_group in branch1_optim.param_groups:
            param_group['lr'] = param_group['lr'] * math.exp(-decay_factor)
        for param_group in branch2_optim.param_groups:
            param_group['lr'] = param_group['lr'] * math.exp(-decay_factor)
    
    return branch1_optim, branch2_optim
