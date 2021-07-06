
def get_branch_params(model):

    shared_params = []
    branch1_params = []
    branch2_params = []

    params = model.state_dict()

    for key, value in params.items():
        if 'branch1' in str(key):
            branch1_params.append(params[key])
        elif 'branch2' in str(key):
            branch2_params.append(params[key])
        else:
            shared_params.append(params[key])

    return shared_params, branch1_params, branch2_params


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


def congestion_scheduler(epoch, min_epoch, decay_factor, mult_factor, max_lr, shared_optim, branch1_optim, branch2_optim, prior_shared_params, prior_branch1_params, prior_branch2_params, branch1_acc, branch2_acc, condition):
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

    if branch1_cond and epoch > min_epoch:
        for param_group in branch1_optim.param_groups:
            param_group['lr'] = min(param_group['lr'] * mult_factor, max_lr)    
        for param_group in branch2_optim.param_groups:
            param_group['lr'] =  param_group['lr'] * math.exp(-decay_factor)
            param_group['params'] = prior_branch2_params
        for param_group in shared_optim.param_groups:
            param_group['lr'] =  param_group['lr'] * math.exp(-decay_factor)
            param_group['params'] = prior_shared_params

    elif branch2_cond and epoch > min_epoch:
        for param_group in branch1_optim.param_groups:
            param_group['lr'] =  param_group['lr'] * math.exp(-decay_factor)
            param_group['params'] = prior_branch1_params
        for param_group in branch2_optim.param_groups:
            param_group['lr'] = min(param_group['lr'] * mult_factor, max_lr)
        for param_group in shared_optim.param_groups:
            param_group['lr'] =  param_group['lr'] * math.exp(-decay_factor)
            param_group['params'] = prior_shared_params
    
    else:
        for param_group in branch1_optim.param_groups:
            param_group['lr'] = param_group['lr'] * math.exp(-decay_factor)
        for param_group in branch2_optim.param_groups:
            param_group['lr'] = param_group['lr'] * math.exp(-decay_factor)
        for param_group in shared_optim.param_groups:
            param_group['lr'] =  param_group['lr'] * math.exp(-decay_factor)
    
    return shared_optim, branch1_optim, branch2_optim