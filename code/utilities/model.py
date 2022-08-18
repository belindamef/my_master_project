import numpy as np
import copy as cp


class Model:
    """A class used to represent the behavioral model

    Attributes
    ----------
    agent : object
        Object of class agent
    a_t : array_like
        Action value in trial t
    """

    def __init__(self):

        # Initialize empty attribute to embed agent object of class Agent
        self.agent = None

        # Initialize action attribute
        self.a_t = np.full(1, np.nan)  # agent action

    def return_action(self):
        """This function returns the action value given agent's decision."""
        self.a_t = cp.deepcopy(self.agent.d)  # probability action given decision of 1
