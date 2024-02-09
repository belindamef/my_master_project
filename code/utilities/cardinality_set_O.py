"""
A script to evaluate the cardinality of the constrained observation set 
of the treasure hunt game. 

Author - Dirk Ostwald
"""
import numpy as np                                                              # numpy
from math import factorial as fact                                              # factorial

# -----------------------------------------------------------------------------
def bino(n,k):                                                                          
    """
    This function evaluates the binomial coefficient.  
    Inputs
            n   : set size
            k   : selection size
    Outputs
            n over k
    """
    return fact(n) // fact(k) // fact(n-k)                                      # binomial coefficient evaluation

def cOb(n_c, n_h):

    """
    This function evaluates the cardinality of the subset of the unconstrained 
    observation set that violates the condition that the number of blue cells
    is upper bounded by the number of hiding spots

    Inputs
        n_c     : number of grid world cells
        n_h     : number of hiding spots

    Outputs
        n_ob    : cardinality of the set of interest
    """
    n_ob        = 0                                                             # initialization
    for k_b in np.arange(n_h+1,n_c+1):                                          # blue cell number iterations
        n_g     = 0                                                             # possible number of complementary gray cells
        for k_g in np.arange(0,n_c-k_b + 1):                                    # gray cell number iterations    
            n_g = n_g + bino(n_c - k_b, k_g)                                    # number of gray cell scenarios             
        n_ob = n_ob + (bino(n_c,k_b)*n_g)                                       # O_b cardinality update
    return n_ob                                                                 # output specification

def cOg(n_c, n_h):

    """
    This function evaluates the cardinality of the subset of the unconstrained 
    observation set that violates the condition that the number of gray cells
    is upper bounded by the number of grid cells minus the number of  of hiding
    spots.

    Inputs
        n_c     : number of grid world cells
        n_h     : number of hiding spots

    Outputs
        n_og    : cardinality of the set of interest
    """
    n_og        = 0                                                             # initialization
    for k_g in np.arange(n_c-n_h+1,n_c+1):                                      # gray cell number iterations
        n_b     = 0                                                             # possible number of complementary blue cells
        for k_b in np.arange(0,n_c-k_g+1):                                      # gray cell number iterations           
            n_b = n_b + bino(n_c - k_g, k_b)                                    # number of blue cell scenarios 
        n_og = n_og + (bino(n_c,k_g)*n_b)                                       # O_g cardinality update
    return n_og                                                                 # output specification

# -----------------------------------------------------------------------------
# simulation set upt
d       = 2                                                                     # square grid world dimension
n_c     = d ** 2                                                                # number of cells   
n_h     = 1                                                                     # number of hiding spots
C       = [0,1,2]                                                               # cell colors (0 black - not investigate, 1 gray - investigated, not hiding spot, 2 blue - investigated hiding spot)
n_oc    = len(C) ** n_c                                                         # cardinality of the unconstrained observation set

# analytical solution
n_ob    = cOb(n_c, n_h)                                                         # cardinality of violation set O_b    
n_og    = cOg(n_c, n_h)                                                         # cardinality of violation set O_g
n_o     = n_oc - n_ob - n_og                                                    # cardinality of the constrained observation set 

# iterative solution
O       = np.empty((0,n_c))                                                     # constrained observation set initialization
n_obi   = 0                                                                     # cardinality of violation set O_b by iteration initialization   
n_ogi   = 0                                                                     # cardinality of violation set O_g by iteration initialization   
for c1 in C:                                                                    # cell 1
     for c2 in C:                                                               # cell 2
         for c3 in C:                                                           # cell 3    
             for c4 in C:                                                       # cell 4
                 o      = np.array([c1,c2,c3,c4])                               # observation of interest        
                 n_o_b  = np.count_nonzero(o == 2)                              # number of blue cells
                 n_o_g  = np.count_nonzero(o == 1)                              # number of gray cells     
                 if n_o_b > n_h:                                    
                    n_obi = n_obi + 1                                           # cardinality of blue cell violation set update                                    
                 if n_o_g > n_c - n_h:
                    n_ogi = n_ogi + 1                                           # cardinality of blue cell violation set update   
                 if n_o_g <= n_c - n_h and n_o_b <= n_h:                        # observation set constraint
                     O  = np.vstack([O, o])                                     # constrained observation set concatenation               
n_oi    = O.shape[0]                                                            # cardinality of the observation set by iteration
print(n_ob, n_obi, n_og, n_ogi, n_o, n_oi)                                      # analytical and iterative cardinalities comparison  


