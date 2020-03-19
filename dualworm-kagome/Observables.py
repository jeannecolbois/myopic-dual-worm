
# coding: utf-8

# ### Functions representing observables

# In[ ]:


import numpy as np


# In[ ]:


def energy(stlen, state, en_state, spinstate, s_ijl, ijl_s, **kwargs):
    return en_state/stlen


# In[ ]:


def magnetisation(stlen, state, en_state, spinstate, s_ijl, ijl_s,**kwargs):
    M = np.sum(spinstate)
    #print(M/stlen)
    return abs(M/stlen)

# In[ ]:

#def allcorrelations(stlen, state, en_state, spinstate, s_ijl, ijl_s):
#    return spinstate
#
def centralcorrelations(stlen, state, en_state, spinstate, s_ijl, ijl_s,**kwargs):
    L = np.sqrt(stlen/9)
    ref_spin = [spinstate[ijl_s[(L, L, 0)]], spinstate[ijl_s[(L, L, 1)]], spinstate[ijl_s[(L, L, 2)]]]
    central_corr = [ref_spin[0]*spinstate, ref_spin[1]*spinstate, ref_spin[2]*spinstate]
    
    #print(central_corr[0][ijl_s[(L,L,1)]])
    #print(central_corr[0][ijl_s[(L,L,2)]])
    return np.array(central_corr) 


def si(stlen, state, en_state, spinstate, s_ijl, ijl_s,**kwargs):
    return np.array(spinstate, dtype = 'int8')

def firstcorrelations(stlen, state, en_state, spinstate, s_ijl, ijl_s, nnlists = [], m = 0, **kwargs):
    mnow = [[sum([spinstate[s1] for (s1, s2) in nnlist])/len(nnlist),
             sum([spinstate[s2] for (s1, s2) in nnlist])/len(nnlist)]
            for nnlist in nnlists]
    firstcorr = [sum([spinstate[s1]*spinstate[s2] for (s1,s2) in nnlist])/len(nnlist) - mnow[index][0]*mnow[index][1] for index, nnlist in enumerate(nnlists)]
    #print(firstcorr)
    return np.array(firstcorr)

def charges(stlen, state, en_state, spinstate, s_ijl, ijl_s,c2s = [], csign= [], **kwargs):
    cvals = np.array([csign[c]*(spinstate[s1]+spinstate[s2]+spinstate[s3]) for c, (s1,s2,s3) in enumerate(c2s)], dtype = 'int8')
    return cvals
    
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
