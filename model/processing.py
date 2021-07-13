def get_branch_params(model):
    '''
    A function to return the parameters for each branch
     of the model.
    Will be used for the separate optimizers when implementing 
    the congestion control protocol, this will allow a differing
    learning rate on each branch.
    '''

    branch1_params = []
    branch2_params = []
    shared_params = []

    params = model.state_dict()

    for key, value in params.items():
        if 'branch1' in str(key):
            branch1_params.append(params[key])
        elif 'branch2' in str(key):
            branch2_params.append(params[key])
        else:
            shared_params.append(params[key])

    return shared_params, branch1_params, branch2_params


def get_branch_params_list(model):
    '''
    A function to return the parameters for each branch
     of the model.
    Will be used for the separate optimizers when implementing 
    the congestion control protocol, this will allow a differing
    learning rate on each branch.
    '''

    shared_params = list(model.conv1.parameters()) + list(model.bn1.parameters()) + list(model.relu.parameters()) + list(model.maxpool.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters())

    branch1_params = list(model.branch1layer3.parameters()) + list(model.branch1layer4.parameters()) + list(model.avgpool.parameters()) + list(model.branch1fc.parameters())

    branch2_params = list(model.branch2layer3.parameters()) + list(model.branch2layer4.parameters()) + list(model.avgpool.parameters()) + list(model.branch2fc.parameters())    

    return shared_params, branch1_params, branch2_params