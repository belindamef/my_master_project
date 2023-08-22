"""Module containing the Behavioral model class"""
import copy as cp
import numpy as np
from .agent import Agent


class BehavioralModel:
    """A class used to represent the behavioral beh_model

    Attributes
    ----------
    agent : object
        Object of class agent
    a_t : array_like
        Action value in trial t
    """
    p_a_giv_h: np.ndarray  # likelihood fucntion of action giv history and tau
    rvs: np.ndarray
    log_likelihood: float = np.nan
    action_t = np.nan  # agent action

    def __init__(self, tau: float, agent_object: Agent):

        self.agent: Agent = agent_object
        self.tau = tau  # decision noice parameter

    def eval_p_a_giv_tau(self):
        """Evaluate conditional probability distribution of actions given the
        history of actions and observations and tau
        aka. likehood of this tau"""
        self.p_a_giv_h = np.exp((1 / self.tau) * self.agent.valence_t) / sum(
            np.exp((1 / self.tau) * self.agent.valence_t))

    def eval_rvs(self):
        """Evaluate action according to sample from multinomial distribution
        TODO what does rvs stand for"""
        rng = np.random.default_rng()
        self.rvs = rng.multinomial(1, self.p_a_giv_h)

    def return_action(self):
        """This function returns the action value given agent's decision."""
        # probability action given decision of 1
        if (np.isnan(self.tau) or self.tau == 0):
            self.action_t = cp.deepcopy(self.agent.decision_t)

        else:
            self.eval_p_a_giv_tau()
            self.eval_rvs()
            action_index = self.rvs.argmax()
            self.action_t = self.agent.a_s1[action_index]

        return self.action_t

    def eval_p_a_giv_h_this_action(self, this_action):
        """Evaluate the conditional probability of this action given the
        history of actions and observations and tau aka. log likelihood of this
          tau"""
        self.log_likelihood = float(np.log(
            self.p_a_giv_h[
                np.where(self.agent.a_s1 == this_action)[0][0]]
            ))
