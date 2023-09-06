#!/usr/bin/env python3
"""
Created on Thu Jul 28 09:59:40 2022

This script contains the implementation of the Bayesian model comparison 
function.

@author: Franziska Usée, Dirk Ostwald
"""

# import dependencies
import numpy as np                                                              # numerical operations
from sklearn.utils import Bunch                                                 # container object exposing keys as attributes 
from scipy.special import digamma as psi                                        # digamma function
from scipy.special import gammaln                                               # log Gamma function
from scipy.stats import dirichlet                                               # Dirichlet distribution functionality

def abm_bmc(bmc): 
    
    """
    This function performs Bayesian model comparison based on a 
    participant x models array of maximized log likelihood values.
    
    Input:
        bmc: Bayesian model comparison structure with fields
            .mll    : participants x models array of maximized log likelihoods
            .n      : participants x models array of observation numbers
            .k      : participants x models array of free parameter numbers
        
    Output:
        bmc: input structure with additional field
            .mll_avg : 1 x models array of maximized log likelihood averages
            .bic     : participants x models array of BICs
            .bic_sum : 1 x models array of BIC sum values
            .win     : p x 1 array indexing participant-specific winning models
            .gmi     : group model inference structure
            .phi     : 1 x models array of protected exceedance probabilities
            .p_eta_0 : posterior probability of null hypothesis ("Bayes omnibus risk")
    """
  
    # average maximimized log likelihood evaluation
    bmc.mll_avg     = bmc.mll.mean(axis = 0)                                    # group average maximized log likelihood
    
    # Bayesian information criterion (BIC) evaluation
    bmc.bic         = bmc.mll - 0.5 * bmc.k * np.log(bmc.n)                     # participant-specific BICs
    bmc.bic_sum     = bmc.bic.sum(axis = 0)                                     # group BIC sum 
    
    # BIC-based winning model evaluation
    bmc.win         = np.zeros([bmc.bic_sum.shape[0]])                          # group winning model indicator initialization
    bmc.win         = np.zeros([int(np.unique(bmc.n))])                         # participant-specific winning model indicator initialization
    bmc.win         = [np.argmax(i) for i in bmc.bic]                           # participant-specific winning model indicator evaluation
    
    # group model inference model-based protected exceedance probabilites evaluation
    gmi             = Bunch()                                                   # bunch initialization
    gmi.lme         = bmc.bic                                                   # log model evidence approximations = BICs
    gmi.n           = gmi.lme.shape[0]                                          # number of participants  
    gmi.m           = gmi.lme.shape[1]                                          # number of models 
    gmi.alpha       = np.ones(gmi.m)                                            # Dirichlet prior parameter
    gmi.delta       = 1e-12                                                     # variational inference algorithm convergence criterion
    gmi             = gmi_via(gmi)                                              # variational inference algorithm
    gmi             = gmi_mce(gmi)                                              # Monte Carlo estimate of the model exceedance probability for \eta = 1
    gmi             = gmi_bor(gmi)                                              # posterior probability of \eta = 0 ("Bayes omnibus risk")
    gmi             = gmi_phi(gmi)                                              # protected model exceedance probabilities    
     
    # output specification
    bmc.gmi         = gmi                                                       # entire group model inference structure
    bmc.phi         = gmi.phi                                                   # protected exceedance probabilities
    bmc.p_eta_0     = gmi.p_eta_0                                               # posterior probability of \eta = 0
 
    return bmc

def gmi_via(gmi):
    
    """
    This function implements a variational inference algorithm for \eta = 1 
    
    Input:
        gmi: group model inference structure  with fields
            .lme        : n x m log model evidence approximations array
            .n          : scalar number of participants  
            .m          : scalar number of models 
            .alpha      : Dirichlet prior parameter
            .delta      : scalar algorithm convergence criterion
    
    Output
        gmi: input structure with additional fields
            .pi_q_i     : converged variational Multinoulli parameters for \eta = 1
            .alpha_q    : converged variational Dirichlet parameters 
            .E_q_pi     : converged variational Dirichlet expectation
            .c          : number of algorithm iterations at convergence       
    """

    # algorithm initialization
    n           = gmi.n                                                         # number of participants
    m           = gmi.m                                                         # number of models
    k           = 0                                                             # iteration counter 
    lme         = gmi.lme                                                       # observed approximated log model evidences \ln p(a_i|\mu_i)
    delta       = gmi.delta                                                     # convergence criterion
    alpha       = gmi.alpha                                                     # Dirichlet prior parameters
    conv        = False                                                         # algorithm convergence flag

    # variational inference algorithm initialization k := 0
    alpha_q     = alpha                                                         # q^{(0)}(\pi) := p_\alpha(\pi)  
    pi_q_i      = np.full([n,m], np.nan)                                        # participant specific variational distributions on \mu_i 

    # algorithm iterations
    while not conv:
        k       = k + 1                                                         # iteration counter update k = 1,2,...

        # variational distribution updates q^{k+1}(\mu_i) for i = 1,...,n
        for i in range(n):                                                      # participant iterations            
            for j in range(m):                                                  # model indicator vector entry iterations
                pi_q_i[i,j] = np.exp(lme[i,j]                                   # q(\mu_i) \propto \exp(ln p(a_i|\mu_i)
                                     + psi(alpha_q[j])                          #                       + \psi(\alpha_j^q) 
                                     - psi(alpha_q.sum()))                      #                        - psi(\alpha_0^q)) 
            pi_q_i[i,:] = pi_q_i[i,:]/pi_q_i[i,:].sum()                         # normalization over \mu_i \in E

        # variational distribution update q^{k+1}(\pi)
        beta_j      = pi_q_i.sum(axis = 0)                                      # \beta_j    = \sum_{i=1}^n \pi^{q}_{ij} 
        alpha_q_pre = alpha_q                                                   # alpha_q pre update
        alpha_q     = alpha + beta_j                                            # \alpha_j^q  = \alpha_j + \beta_j  
        eps         = np.linalg.norm(alpha_q - alpha_q_pre)                     #  Euclidean norm of parameter vector update 

        # covergence assessment
        if eps < delta:                                                         # convergence criterion check
            conv = True                                                         # convergence

    # output specification
    gmi.c         = k                                                           # number of algorithm iterations
    gmi.alpha_q     = alpha_q                                                   # converged variational Dirichlet parameters
    gmi.E_q_pi      = gmi.alpha_q/np.sum(gmi.alpha_q)                           # converged variational Dirichlet expectation
    gmi.pi_q_i      = pi_q_i                                                    # converged variational Multinoulli parameters
    
    return gmi

def gmi_mce(gmi):

    """
    This function evaluates the Monte Carlo estimate of the model exceedance
    probabilities for \eta = 1

    Inputs
        gmi: group model inference model structure with required fields
            .alpha_q    : converged variational Dirichlet parameters 
            .m          : number of models
    Outputs
        gmi: input structure with additional fields
            .phi_eta_1  : Monte Carlo exceedance probability estimates for \eta = 1
      
    """ 
    m           = gmi.m                                                         # number of models
    s           = int(1e6)                                                      # number of samples for exceedance probability approximation  
    alpha_q     = gmi.alpha_q                                                   # Dirichlet distribution parameters
    pi_k        = dirichlet.rvs(alpha_q, size = s)                              # \pi^{(1)}, ..., \pi^{(s)} \sim Dir(\alpha^q)
    pi_k_max    = np.argmax(pi_k, axis = 1)                                     # \max{\pi^{(k)}_1, ..., \pi^{(k)}_m} for all k = 1,...,s                               
    sI_j        = np.full([m,], np.nan)                                         # indicator function sums initialization       
    for j in range(m):                                                          # model index iterations    
        sI_j[j] = pi_k_max[pi_k_max == j].size                                  # \sum_{k=1}^s \mathbb{I}_j(\pi^{(k)}): number of times that \pi_j = \max{\pi_1, ..., \pi_m}
    phi_eta_1   = (1/s)*sI_j                                                    # Monte Carlo estimate of exceedance probabilities
    
    # output specification
    gmi.phi_eta_1 = phi_eta_1                                                   # Monte Carlo estimate model exceedance probabilities for \eta = 1
    return gmi

def gmi_bor(gmi):
    
    """
    This function evaluates the Bayes omnibus risk, i.e., posterior probability 
    of \eta = 0, based on Appendix 2 in  Rigoux et al. (2014) "Bayesian model 
    selection for group studies - revisited" NeuroImage 84 on page 984.

    Inputs
        gmi: group model inference structure with required fields
            .lme        : n x m log model evidence array
            .n          : scalar number of participants  
            .m          : scalar number of models 
            .pi_q_i     : participant-specific variational model distributions
            .alpha_q    : converged variational Dirichlet parameter
    
    
    Output
        gmi: group model inference structure with additional field
            .E0         : ELBO for \eta = 0
            .E1         : ELBO for \eta = 1
            .p_eta_0    : approximation of the posterior probability of \eta = 0
    """
    # data extraction
    lme         = gmi.lme                                                       # observed log model evidences
    m           = gmi.m                                                         # number of models
    pi_q_i      = gmi.pi_q_i                                                    # participant-specific variational distributions 
    alpha       = gmi.alpha                                                     # prior Dirichlet parameter
    alpha_q     = gmi.alpha_q                                                   # converged variational Dirichlet parameter
    
    # ELBO for \eta = 0
    W           = batch_softmax(lme, axis = 1)                                  # multinoulli parameters exp(ln(y_i|\mu_i))/sum_{\mu_i in E} exp(ln(y_i|\mu_i))
    E0          = np.sum(W*(lme - np.log(m) - np.ma.log(W)))                    # ELBO for \eta = 0

    # ELBO for \eta = 1
    T1          = gammaln(alpha.sum())-np.sum(gammaln(alpha))+np.sum((alpha-1)*(psi(alpha_q)-psi(alpha_q.sum())))
    T2          = np.sum(pi_q_i*((psi(alpha_q) - psi(alpha_q.sum())))) 
    T3          = np.sum(pi_q_i*lme)
    T4          = gammaln(alpha_q.sum())-np.sum(gammaln(alpha_q))+np.sum((alpha_q-1)*(psi(alpha_q)-psi(alpha_q.sum())))
    T5          = np.sum(pi_q_i*np.ma.log(pi_q_i))
    E1          = T1 + T2 + T3 - T4 - T5

    # approximation of the posterior probability of \eta = 0 
    p_eta_0     = 1/(1 + np.exp(E1 - E0))                                     

    # output specification
    gmi.E0      = E0                                                            # ELBO for \eta = 0
    gmi.E1      = E1                                                            # ELBO for \eta = 1
    gmi.p_eta_0 = p_eta_0                                                       # \approx P(\eta = 0|y_{1:n})

    return gmi

def gmi_phi(gmi):

    """
    This function evaluates the protected model exceedance probabilities 
    Inputs
        gmi: group model inference structure with required fields
            .mce       : Monte Carlo model exceedance probabilities for \eta = 1 estimates
            .p_eta_0   : posterior probability of \eta = 0 
            .m         : scalar number of models 
    
    Output
        gmi: group model inference structure with additional field
            .phi        : protected exceedance probabilities 

    """
    # data extraction
    p_eta_0     = gmi.p_eta_0                                                   #  posterior probability of \eta = 0 
    phi_eta_0   = 1/gmi.m                                                       # model exceedance probabilities for \eta = 0
    phi_eta_1   = gmi.phi_eta_1                                                 # model exceedance probabilities for \eta = 1
    phi         = phi_eta_0*p_eta_0 + phi_eta_1*(1-p_eta_0)                     # protected exceedance probabilities

    # output specification
    gmi.phi     = phi                                                           

    return gmi 


def reduce_then_tile(X, f, axis=1):
    """ Computes some reduction function over an axis, then tiles that vector 
    to create matrix of original size

    Arguments:

        X: `ndarray((n, m))`. Matrix.
        f: `function` that reduces data across some axis (e.g. `np.sum()`, `np.max()`)
        axis: `int` which axis the data should be reduced over (only goes over 2 axes for now)

    Returns:res

        `ndarray((n, m))`

    Examples:

    Here is one way to compute a softmax function over the columns of `X`, for each row.

    ```
    import numpy as np
    X = np.random.normal(0, 1, size=(10, 3))**2
    max_x = reduce_then_tile(X, np.max, axis=1)
    exp_x = np.exp(X - max_x)
    sum_exp_x = reduce_then_tile(exp_x, np.sum, axis=1)
    y = exp_x/sum_exp_x
    ```

    """
    y = f(X, axis=axis)
    if axis==1:
        y = np.tile(y.reshape(-1, 1), [1, X.shape[1]])
    elif axis==0:
        y = np.tile(y.reshape(1, -1), [X.shape[0], 1])
    return y

def batch_softmax(X, axis=1):
    """ Computes the softmax function for a batch of samples

    $$
    p(\mathbf{x}) = \\frac{e^{\mathbf{x} - \max_i x_i}}{\mathbf{1}^\\top e^{\mathbf{x} - \max_i x_i}}
    $$

    Arguments:

        x: Softmax logits (`ndarray((nsamples,nfeatures))`)

    Returns:

        Matrix of probabilities of size `ndarray((nsamples,nfeatures))` such that sum over `nfeatures` is 1.
    """
    xmax = reduce_then_tile(X, np.max, axis=axis)
    expx = np.exp(X - xmax)
    y = expx/reduce_then_tile(expx, np.sum, axis=axis)
    return y

