import numpy as np
import copy as cp


class Model:
    """
    TODO
    """
    def __init__(self, m_init):
        """
        This function is the instantiation operation of the model class.

        Input
            m_init        : model object
                  .agent  : agent object
                  .task   : task object

        Output
            m_init        : input object with additional attributes
                  .a      : action after evaluating action noise
        """
        # Define structural component
        self.agent = None

        # Define dynamic model components
        self.a = np.full(1, np.nan)  # agent action

    # Model functions
    # ------------------------------------------------------------------------
    def return_action(self):
        """
        This function returns the agent's action given agent's decision.

        Input
            self         : model object
                .a       : action
                .agent.dim : agent object decision
        """
        self.a = cp.deepcopy(self.agent.d)  # probability action given decision with prob of 1
