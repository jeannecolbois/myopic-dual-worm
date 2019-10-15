
# coding: utf-8

# ### Functions representing observables

# In[ ]:


import numpy as np


# In[ ]:


def energy(state, en_state, spinstate, s_ijl):
    return en_state/len(s_ijl)


# In[ ]:


def magnetisation(state, en_state, spinstate, s_ijl):
    M = 0
    for s in spinstate:
        M += s
    return abs(M/len(s_ijl))


# In[ ]:


def A3(state, en_state, spinstate, s_ijl, ijl_s):
    magn = np.array([0, 0, 0])
    for s, (i, j, l) in enumerate(s_ijl):
        magn[l] += spinstate[s]
    magn = (magn / (len(s_ijl)/3)) ** 2
    A3magn = magn.sum(0) / 3
    return A3magn


# In[ ]:


def A9(state, en_state, spinstate, s_ijl, ijl_s):
    magn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for s, (i, j, l) in enumerate(s_ijl):
        subl = 3*((i + 2*j)%3) + l;
        magn[subl] += spinstate[s]
    magn = (magn / (len(s_ijl)/9)) ** 2
    A9magn = magn.sum(0) / 9
    return A9magn


# In[ ]:


def subkag(state, en_state, spinstate, s_ijl, ijl_s):
    magn = np.array([0, 0, 0])
    for s, (i, j, l) in enumerate(s_ijl):
        alpha = (i + 2 * j)%3
        subl = (alpha + 2 * (l%2))%3
        magn[subl] += spinstate[s]
    magn = (magn / (len(s_ijl)/3)) ** 2
    subkag = magn.sum(0)/3
    return subkag


# In[ ]:

def allcorrelations(state, en_state, spinstate, s_ijl):
    return spinstate

def centralcorrelations(state, en_state, spinstate, s_ijl):
    L = np.sqrt(len(s_ijl)/9)
    ref_spin = [0, 0, 0]
    
    corr = spinstate
    for s, (i,j,l) in enumerate(s_ijl):
        if i == L and j == L:
            ref_spin[l] = spinstate[s]
    central_corr = [(ref_spin[0]*corr).tolist(), (ref_spin[1]*corr).tolist(), (ref_spin[2]*corr).tolist()]
    
    return np.array(central_corr) 


def si(state, en_state, spinstate, s_ijl):
    si = np.zeros(len(s_ijl))
    for s, (i, j, l) in enumerate(s_ijl):
        si[s] = spinstate[s]
    return si
