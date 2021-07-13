import torch 

def congestion_avoid(model, optimizer, branch1_acc, branch2_acc, condition, branch_one_grads, branch_two_grads, roll_epochs):

    branch1_cond = branch1_acc < condition * branch2_acc
    branch2_cond = branch2_acc < condition * branch1_acc

    lr = optimizer.param_groups()[0]['lr']

    if branch1_cond:
        print('Branch 1 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                value += roll_epochs * lr * branch_two_grads[name]

    elif branch2_cond:
        print('Branch 2 condition has been met .....')
        for name, value in model.named_parameters():
            with torch.no_grad():
                value += roll_epochs * lr * branch_one_grads[name]
    
    else:
        print('No condition is met .....')

    return optimizer, model