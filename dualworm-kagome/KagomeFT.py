# In[1]:


## Last update 08.01.2020
# Author : Jeanne Colbois
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
        #exponent1 = 2 * np.pi * k1 / L * ((i) + vec_l[l][0])
        #exponent2 = 2 * np.pi * k2 / L * ((j) + vec_l[l][1])
        sum_at_q += f[s] * np.exp(- 1j * (exponent1 + exponent2))
    return sum_at_q





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


def PBCStrctFact(L, sconf, ijl_sconfig, xy_m1m2 = np.zeros((2,2)), subtractm = True, centered = False, a = 2, **kwargs):
    '''
        Computes the full structure factor associated with a spin config,
        with PBC imposed on a kagome system of size L (i.e. computes
        the structure factor of a sub-system if the config doesn't
        have the right shape).
        Inputs:
        - L (system size to use)
        - sconf: sconf[s] = +- 1
        - ijl_sconfig: mapping from (i,j,l) values to the corresponding
        s index for the configuration (=== ijl_s if the system is the usual
        lattice structure)
        - s_pos: actual position for spin s in (x,y) cartesian coordinates
        - subtractm : True for connected correlations, false for disconnected
        ones
        - xy_m1m2: matrix to apply to a vector in (x,y) cartesian coordinates
        to get the vector in (m1,m2) lattice basis coordinates
    '''
    # spin site table:
    (s_ijl, ijl_s) = kf.createspinsitetable(L)
    nspins = len(s_ijl)
    N = np.sqrt((nspins**2)) # normalization for the FT
    
    s_pos, ijl_pos = kf.reducedgraphkag(L, s_ijl, ijl_s)
    
    # super lattice
    n1, n2, Leff, S = kf.superlattice(L)
    
    # list of neighbours:
    listnei = [(0, 0), (0, 1), (1, 0), (-1, 1),
               (-1, 0), (0, -1),(1, -1)]
    

    m = 0
    print("subtractm = {0}".format(subtractm))
    if subtractm:
        for s1 in range(nspins):
            (i1,j1,l1) = s_ijl[s1]
            pos1 = s_pos[s1]
            vals1 = sconf[ijl_sconfig[(i1,j1,l1)]]
            
            m += vals1/nspins
    
    #StrctFactRes = np.zeros((q_k1k2.shape[0],3, 3), dtype = 'complex128')
    print("centered = ", centered)
    if not centered:
        correlations = [[0 for s2 in range(s1+1, nspins)]for s1 in range(nspins)]
        for s1 in range(nspins):
            (i1,j1,l1) = s_ijl[s1]
            vals1 = sconf[ijl_sconfig[(i1,j1,l1)]]
            
            for s2 in range(s1+1, nspins):
                (i2,j2,l2) = s_ijl[s2]
                vals2 = sconf[ijl_sconfig[(i2,j2,l2)]]
                correlations[s1][s2] = np.asscalar(vals1*vals2 - m**2)
                # m is zero if not subtractm
    else:
        for s1 in range(3):
            (i1,j1,l1) = s_ijl[s1]
            vals1 = sconf[ijl_sconfig[(i1,j1,l1)]]
            
            for s2 in range(s1+1, nspins):
                (i2,j2,l2) = s_ijl[s2]
                vals2 = sconf[ijl_sconfig[(i2,j2,l2)]]
                correlations[s1][s2] = np.asscalar(vals1*vals2 - m**2)
                # m is zero if not subtractm
               
            
    StrctFactRes = StrctFact(L, correlations, centered = centered,\
                             s_ijl = s_ijl, ijl_s = ijl_s, s_pos = s_pos,\
                             ijl_pos = ijl_pos, n1 = n1, n2 = n2, Leff = Leff,\
                             S = S, xy_m1m2 = xy_m1m2, nsites = nspins)
                             
    return StrctFactRes, m


def StrctFact(L, correlations, centered = True, srefs = range(3), s_ijl =[],\
              ijl_s = [], s_pos = [], ijl_pos = [], n1 = [],n2 = [], Leff = 0, S = [],\
             q_k1k2= [], k1k2_q = {}, xy_m1m2 = [], nsites = 0, a = 2, **kwargs):
    '''
        Given correlations of a function, computes the associated structure
        factor on the lattice with size L.
        - centered: if the correlations are relatively to three sites only;
        if centered is False the sum over all the possible centers is made.
        - srefs: if centered is True, srefs gives the indices of the three
        reference spins (by default, 0,1,2)
    '''
    
    ## Computing the lattice structure:
    if not s_ijl or nsites == 0:
        (s_ijl, ijl_s) = kf.createspinsitetable(L)
        nsites = len(s_ijl)
        N = np.sqrt((nsites**2)) # normalization for the FT
    if not s_pos:
        s_pos, ijl_pos = kf.reducedgraphkag(L, s_ijl, ijl_s)
    
    # super lattice
    if not n1:
        n1, n2, Leff, S = kf.superlattice(L)
    
    ## Computing the reciprocal lattice structure
    # reciprocal site table:
    if not q_k1k2:
        (q_k1k2, k1k2_q) = kdraw.KagomeReciprocal(L)
        q_k1k2 = np.array(q_k1k2)
    
    # list of neighbours:
    listnei = [(0, 0), (0, 1), (1, 0), (-1, 1),
               (-1, 0), (0, -1),(1, -1)]
    
    # xy_m1m2
    if not xy_m1m2 or np.linalg.det(xy_m1m2) == 0:
        xy_m1m2 = (1/a)*np.array([[1, -1/np.sqrt(3)],[0, 2/np.sqrt(3)]])
    
    ## Computing the structure factor associated with the correlations
    StrctFactRes = np.zeros((q_k1k2.shape[0],3, 3), dtype = 'complex128')
    if not centered:
        for s1 in range(nsites):
            (i1,j1,l1) = s_ijl[s1]
            pos1 = s_pos[s1]
            # equivalent positions with PBC
            pos1list = np.array([pos1 + nei[0]*Leff*n1 + nei[1]*Leff*n2
                                 for nei in listnei])
             
            
            for s2 in range(s1+1, nsites):
                (i2,j2,l2) = s_ijl[s2]
                pos2 = s_pos[s2]
                
                # separation
                sep = pos2 - pos1list
                
                # index of minmum distance
                neiid = np.argmin([np.linalg.norm(sep[i]) for i in
                                   range(sep.shape[0])])

                # position difference in (m1, m2) coordinates
                dm1m2 = np.dot(xy_m1m2, sep[neiid]) 

                exponent = 1j * 2 * np.pi * np.dot(q_k1k2, dm1m2)/L
                StrctFactRes[:,l1, l2] +=\
                correlations[s1][s2]*np.exp(exponent)/N
                StrctFactRes[:,l2, l1] +=\
                correlations[s1][s2]*np.exp(-exponent)/N
                
    else:
        for s1 in range(3):
            s1id = srefs[s1]
            (i1,j1,l1) = s_ijl[s1id]
            pos1 = s_pos[s1id]
            # equivalent positions with PBC
            pos1list = np.array([pos1 + nei[0]*Leff*n1 + nei[1]*Leff*n2
                                 for nei in listnei])
            
            for s2 in range(s1+1, nsites):
                (i2,j2,l2) = s_ijl[s2]
                pos2 = s_pos[s2]
                # separation
                sep = pos2 - pos1list
                # index of minmum distance
                neiid = np.argmin([np.linalg.norm(sep[i]) for i in
                                   range(sep.shape[0])])

                # position difference in (m1, m2) coordinates
                dm1m2 = np.dot(xy_m1m2, sep[neiid]) 

                exponent = 1j * 2 * np.pi * np.dot(q_k1k2, dm1m2)/L
                StrctFactRes[:,l1, l2] +=\
                correlations[s1][s2]*np.exp(exponent)/N
                StrctFactRes[:,l2, l1] +=\
                correlations[s1][s2]*np.exp(-exponent)/N    
    
    return StrctFactRes

def OBCStrctFact(ijl, m1m2, sconf, L, subtractm = True, **kwargs):
    '''
        Ad-hoc structure factor computation for a configuration with
        open boundaries (i.e. finite-size support)
        - ijl[s] = (i,j,l)
        - m1m2: config m1m2 mapping
        - sconf: spin configuration
    '''
    (q_k1k2, k1k2_q) = kdraw.KagomeReciprocal(L)
    q_k1k2 = np.array(q_k1k2)
    
    StrctFactRes = np.zeros((q_k1k2.shape[0],3, 3), dtype = 'complex128')
    nspins = len(sconf)
    m = sum(sconf)/sum(abs(sconf))
    if not subtractm:
        m = 0
        
    N = np.sqrt((nspins**2)/2)
    for s1 in range(nspins):
        (i1,j1,l1) = ijl[s1]
        if not l1 == -1:
            for s2 in range(s1+1, nspins):
                # correlation
                c = np.asscalar(sconf[s1]*sconf[s2]-m**2)
                # structure factor computation
                (i2,j2,l2) = ijl[s2]
                if not l2 == -1:
                    exponent = 1j*2 * np.pi * np.dot(q_k1k2, (m1m2[s1]-m1m2[s2]))/L
                    StrctFactRes[:,l1,l2] += c * np.exp(exponent)/N
                    StrctFactRes[:,l2,l1] += c * np.exp(-exponent)/N
    return StrctFactRes, m



#def StrctFact(corr0, corr1, corr2):
#    '''
#        Returns the structure factor linked to the correlation functions corr0, corr1 and corr2, which are respectively
#        centered on (L, L, 0), (L, L, 1) and (L, L, 2).
#    '''
#     ## define the lattice depending on the size of the function we get
#    # size of the function:
#    num_sites = len(corr0) # = 9* L*L
#    # size of the side of the lattice:
#    L = np.sqrt(num_sites/9)
#    L = L.astype(int)
#    # spin site table:
#    (s_ijl, ijl_s) = kf.createspinsitetable(L)
#    # reciprocal site table:
#    (q_k1k2, k1k2_q) = kdraw.KagomeReciprocal(L)
#
#    ## Define the return vector
#    StrctFactRes = np.empty(len(q_k1k2), dtype = 'complex128')
#    StrF0 = np.empty(len(q_k1k2), dtype = 'complex128')
#    StrF1 = np.empty(len(q_k1k2), dtype = 'complex128')
#    StrF2 = np.empty(len(q_k1k2), dtype = 'complex128')
#    vec_l = [[0, 0],[-0.5, 0.5],[-1, 0.5]] #### /!!!!!!\ make sure
#    for q, (k1, k2) in enumerate(q_k1k2):
#        exponent1 = 2 * np.pi * k1 / L * (vec_l[0][0])
#        exponent2 = 2 * np.pi * k2 / L * (vec_l[0][1])
#        StrF0[q] = FTf_at_q(L, k1, k2, corr0, s_ijl, ijl_s, vec_l) * np.exp(- 1j * (exponent1 + exponent2))
#
#        exponent1 = 2 * np.pi * k1 / L * (vec_l[1][0])
#        exponent2 = 2 * np.pi * k2 / L * (vec_l[1][1])
#        StrF1[q] = FTf_at_q(L, k1, k2, corr1, s_ijl, ijl_s, vec_l)* np.exp(- 1j * (exponent1 + exponent2))
#
#        exponent1 = 2 * np.pi * k1 / L * (vec_l[2][0])
#        exponent2 = 2 * np.pi * k2 / L * (vec_l[2][1])
#        StrF2[q] = FTf_at_q(L, k1, k2, corr2, s_ijl, ijl_s, vec_l)* np.exp(- 1j * (exponent1 + exponent2))
#
#    StrctFactRes = StrF0 + StrF1 + StrF2
#    return StrctFactRes, StrF0, StrF1, StrF2
#