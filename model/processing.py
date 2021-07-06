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