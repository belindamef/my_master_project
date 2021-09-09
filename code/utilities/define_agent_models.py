

def define_agent_model(agent_model):
    """
    This function returns an agent initialization object

    Parameters
    ----------
    agent_model : str

    Returns
    -------
    a_init : object
        Object of class AbmStructure
    """
    # Control models
    if agent_model in ['C1', 'C2', 'C3']:
        bayesian = False
        exploring = False

    # Bayesian models
    elif agent_model == 'A1':
        bayesian = True
        exploring = False

    # Bayesian models using explorative strategy
    elif agent_model in ['A2', 'A3']:
        bayesian = True
        exploring = True

    return bayesian, exploring
