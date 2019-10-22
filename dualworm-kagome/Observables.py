
# coding: utf-8

# ### Functions representing observables

# In[ ]:


import numpy as np


# In[ ]:


def energy(stlen, state, en_state, spinstate, s_ijl, ijl_s):
    return en_state/stlen


# In[ ]:


def magnetisation(stlen, state, en_state, spinstate, s_ijl, ijl_s):
    M = np.sum(spinstate)
    return abs(M/stlen)


# In[ ]:


#def A3(stlen, state, en_state, spinstate, s_ijl, ijl_s):
#    magn = np.array([0, 0, 0])
#    for s, (i, j, l) in enumerate(s_ijl):
#        magn[l] += spinstate[s]
#    magn = (magn / (stlen/3)) ** 2
#    A3magn = magn.sum(0) / 3
#    return A3magn
#
#
## In[ ]:
#
#
#def A9(stlen, state, en_state, spinstate, s_ijl, ijl_s):
#    magn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
#    for s, (i, j, l) in enumerate(s_ijl):
#        subl = 3*((i + 2*j)%3) + l;
#        magn[subl] += spinstate[s]
#    magn = (magn / (stlen/9)) ** 2
#    A9magn = magn.sum(0) / 9
#    return A9magn
#
#
## In[ ]:
#
#
#def subkag(stlen, state, en_state, spinstate, s_ijl, ijl_s):
#    magn = np.array([0, 0, 0])
#    for s, (i, j, l) in enumerate(s_ijl):
#        alpha = (i + 2 * j)%3
#        subl = (alpha + 2 * (l%2))%3
#        magn[subl] += spinstate[s]
#    magn = (magn / (stlen/3)) ** 2
#    subkag = magn.sum(0)/3
#    return subkag
#

# In[ ]:

#def allcorrelations(stlen, state, en_state, spinstate, s_ijl, ijl_s):
#    return spinstate
#
def centralcorrelations(stlen, state, en_state, spinstate, s_ijl, ijl_s):
    L = np.sqrt(stlen/9)
    ref_spin = [spinstate[ijl_s[(L, L, 0)]], spinstate[ijl_s[(L, L, 1)]], spinstate[ijl_s[(L, L, 2)]]]
    central_corr = [ref_spin[0]*spinstate, ref_spin[1]*spinstate, ref_spin[2]*spinstate]
    
    print(central_corr[0][0])
    return np.array(central_corr) 


def si(stlen, state, en_state, spinstate, s_ijl, ijl_s):
    return spinstate
