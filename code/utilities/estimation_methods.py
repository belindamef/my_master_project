"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters

Author: Belinda Fleischmann
"""
from .simulation_methods import Simulator
import numpy as np

class ParameterEstimator:
    """A class to evaluate Maximum Likelihood estimations of model parameter"""

    def __init__(self, sim_object):
        """

        Parameters
        ----------
        sim_object: Simulator
        """
        self.sim_object = sim_object


    def eval_llh(self, given_tau):
        """Evaluate log_likelihood functions for given simulated dataset and
        tau parameter (decision noise)

        Parameters
        ----------
        given_tau: float
        """
        self.sim_object.tau = given_tau
        sum_llh = self.sim_object.sim_interaction_for_llh()  # block log
        # likelihood evaluation

        # negative log likelihood and number of observations
        # TODO: need negative???
        # -------------------------------------------------------------------------
        #llh_theta = -llh_b.sum()
        #llh.n = n_b.sum()

        # output specification
        return sum_llh


    def brute_force_est(self):
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter based on dataset of one participant.

        Returns
        -------
        ml_estimate: float
        """
        #tau_space = np.linspace(0.01, 2.5, 50)
        tau_space = np.arange(0.5, 2.5, 0.1)
        # tau_space = np.linspace(0.25,2.5,5)
        loglikelihood_function = np.full(len(tau_space), np.nan)

        for i, tau_i in np.ndenumerate(tau_space):  # candidate set iteration
            # tau_i = np.asarray([tau_i])  # array conversion #TODO: necessary?
            loglikelihood_function[i] = self.eval_llh(tau_i)

        # TODO: Figure out if neg or positive loglikelihood values?
        #  --> min vs. max
        ml_estimate = tau_space[np.argmin(loglikelihood_function)]  # MLE estimate
        llh_value = np.asarray([np.min(loglikelihood_function)])  # minimized negative log likelihood function value

        return ml_estimate