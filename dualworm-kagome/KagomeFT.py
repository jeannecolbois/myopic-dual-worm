
# coding: utf-8

# In[1]:


# This script compute the fourier transform of any function defined on the kagome lattice
# i.e.: FTf(q) = sum_cellInLattice sum_siteInCell f(R_cellInLattice + r_siteInCell) exp(-i q dot (R_cell + r_site))
# The function is juste a vector associating to each site of the lattice a value
# It is important that the function respects the s_ijl structure defined below.


# In[2]:


import numpy as np
import KagomeFunctions as kf # library allowing to work on kagome
import KagomeDrawing as kdraw # library allowing to plot kagome and its reciprocal lattice
import matplotlib.pyplot as plt
#import numba


# In[3]:


def spinsiteUsingBC(i, j, l, L):
    ni = i
    nj = j
    nl = l

    if(i + j < L-1):
        ni += L
        nj += L

    return (ni, nj, nl)


# In[4]:


#@numba.jit(nopython = True)
def FTf_at_q(L, k1, k2, f, s_ijl, ijl_s, vec_l):
    sum_at_q = 0
    for s, (i, j, l) in enumerate(s_ijl):
        exponent1 = 2 * np.pi * k1 / L * ((i - L) + vec_l[l][0])
        exponent2 = 2 * np.pi * k2 / L * ((j - L) + vec_l[l][1])
        sum_at_q += f[s] * np.exp(- 1j * (exponent1 + exponent2))
    return sum_at_q


# In[5]:


def KagomeFT(f):
    '''
        Returns the fourier transform of the function f defined on kagome. The result returned is a vector
        which associates the correct FT to each site of the reciprocal lattice.
        f has to be an array associating a value to each site in the kagome lattice.
        To use this function you need to know how to translate the spin indices from an integer to a location,
        and the same for the reciprocal lattice sites.
    '''


    ### LATTICE CHARACTERISTICS

    ## define the lattice depending on the size of the function we get
    # size of the function:
    num_sites = len(f) # = 9* L*L
    # size of the side of the lattice:
    L = np.sqrt(num_sites/9)
    L = L.astype(int)
    # spin site table:
    (s_ijl, ijl_s) = kf.createspinsitetable(L)
    # reciprocal site table:
    (q_k1k2, k1k2_q) = kdraw.KagomeReciprocal(L)

    ## SHAPES OF f AND FTf

    ## Define the return vector
    FTf = np.empty(len(q_k1k2), dtype = 'complex128')


    ### COMPUTE THE FT
    vec_l = [[0, 0],[-0.5, 0.5],[1, 0.5]]
    for q, (k1, k2) in enumerate(q_k1k2):
        FTf[q] = FTf_at_q(L, k1, k2, f, s_ijl, ijl_s, vec_l)

    return FTf


# In[ ]:


def StrctFact(corr0, corr1, corr2):
    '''
        Returns the structure factor linked to the correlation functions corr0, corr1 and corr2, which are respectively
        centered on (L, L, 0), (L, L, 1) and (L, L, 2).
    '''
     ## define the lattice depending on the size of the function we get
    # size of the function:
    num_sites = len(corr0) # = 9* L*L
    # size of the side of the lattice:
    L = np.sqrt(num_sites/9)
    L = L.astype(int)
    # spin site table:
    (s_ijl, ijl_s) = kf.createspinsitetable(L)
    # reciprocal site table:
    (q_k1k2, k1k2_q) = kdraw.KagomeReciprocal(L)

    ## Define the return vector
    StrctFact = np.empty(len(q_k1k2), dtype = 'complex128')
    StrF0 = np.empty(len(q_k1k2), dtype = 'complex128')
    StrF1 = np.empty(len(q_k1k2), dtype = 'complex128')
    StrF2 = np.empty(len(q_k1k2), dtype = 'complex128')
    vec_l = [[0, 0],[-0.5, 0.5],[1, 0.5]]
    for q, (k1, k2) in enumerate(q_k1k2):
        exponent1 = 2 * np.pi * k1 / L * (vec_l[0][0])
        exponent2 = 2 * np.pi * k2 / L * (vec_l[0][1])
        StrF0[q] = FTf_at_q(L, k1, k2, corr0, s_ijl, ijl_s, vec_l) * np.exp(- 1j * (exponent1 + exponent2))

        exponent1 = 2 * np.pi * k1 / L * (vec_l[1][0])
        exponent2 = 2 * np.pi * k2 / L * (vec_l[1][1])
        StrF1[q] = FTf_at_q(L, k1, k2, corr1, s_ijl, ijl_s, vec_l)* np.exp(- 1j * (exponent1 + exponent2))

        exponent1 = 2 * np.pi * k1 / L * (vec_l[2][0])
        exponent2 = 2 * np.pi * k2 / L * (vec_l[2][1])
        StrF2[q] = FTf_at_q(L, k1, k2, corr2, s_ijl, ijl_s, vec_l)* np.exp(- 1j * (exponent1 + exponent2))

    StrctFact = StrF0 + StrF1 + StrF2
    return StrctFact
