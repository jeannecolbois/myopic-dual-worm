
# coding: utf-8

# In[ ]:

import numpy as np
import dimers as dim


# In[ ]:

def createdualtable(L):
    '''
        Creates the table of dual bonds corresponding to a dice lattice of side size L.
        Returns a table identifing an int with the three coordinates of the dual bond and a dictionnary identifying the
        three coordinates with the dual bond's int index. This allows to handle other relations between dual bonds in an
        easier way.
    '''
    d_ijl = [(i, j, l) for i in range(2*L) for j in range (2*L) for l in range(6) if (i+j > L-2) and (i+j < 3*L-1)]
    
    # we need as well a dictionary to associate values of (i,j,l) to the correct index d
    ijl_d = {} # new empty dictionary
    for d, triplet in enumerate(d_ijl): # same as for d in range(d_ijl) triplet = d_ijl[d]
        ijl_d[triplet] = d
    return (d_ijl, ijl_d)


# In[ ]:

def createspinsitetable(L):
    '''
        Creates the table of spin sites corresponding to a dice lattice of side size L.
        Returns a table identifing an int with the three coordinates of the spin site and a dictionnary identifying the
        three coordinates with the spin site's int index. This allows to handle other relations between spin sites in an
        easier way.
    '''
    s_ijl = [(i, j, l) for i in range(2*L) for j in range(2*L) for l in range(3) if (i+j > L-2) and (i+j < 3*L-1)]
    # dictionary
    ijl_s = {}
    for s, triplet in enumerate(s_ijl):
        ijl_s[triplet] = s
    return s_ijl, ijl_s


# In[ ]:

def fixbc(i, j, l, L):
    '''
        For a lattice side size L, this function handles the periodic boundary conditions by returning the corresponding
        value of i, j, l if they match a point which is just outside the borders of the considered cell.
    '''
    if i == 2*L : # bottom right mapped to top left
        i = 0
        j += L
    if j == 2*L: # top mapped to bottom
        i += L
        j = 0
    if i+j == L-2: # bottom left mapped to top right
        i += L
        j += L
    if i+j == 3*L-1: # top right mapped to bottom left
        i -= L
        j -= L
    if j == -1: # bottom mapped to top
        i -= L
        j = 2*L-1
    if i == -1: # top left mapped to bottom right
        i = 2*L-1
        j -= L
    return (i, j, l)


# In[ ]:

def dualbondspinsitelinks(d_ijl, ijl_s, L):
    '''
        For a lattice with side size L, this function  returns two tables:
        > d_2s: for each dual bond, which are the 2spin sites around it.
        > s2_d: for each pair of spin sites nearest to one another, which is the dual bond between them (dictionary)
    '''
    linkedspinsite = [[(0, -1, 1),(1, -1, 2)],
                  [(1, -1, 2),(0, 0, 0)],
                  [(0, 0, 0),(0, 0, 1)],
                  [(0, 0, 1),(0, 0, 2)],
                  [(0, 0, 2),(-1, 0, 0)],
                  [(-1, 0, 0),(0, -1, 1)]]
    # without worrying about periodic BC:
    d_2s = [[(i + linkedspinsite[l][u][0], j + linkedspinsite[l][u][1], linkedspinsite[l][u][2]) for u in range(2)] for (i, j, l) in d_ijl]
    # fix the periodic boundary conditions
    d_2s = np.array([[ijl_s[fixbc(si, sj, sl, L)] for (si, sj, sl) in dimd] for dimd in d_2s], dtype = 'int32')

    s2_d = {}#empty dictionary
    for d, [s1, s2] in enumerate(d_2s):
        s2_d[(s1, s2)] = d
        s2_d[(s2, s1)] = d #make sure that both orders work

    return d_2s, s2_d


# In[ ]:

def spins_dimers_for_update(s_ijl, ijl_s, s2_d, L):
    '''
        Returns a list of spin site indices and a list of dual bond indices. Going through the spins list allows to map the whole
        spin state of the system. The ith dimer lies between the ith and ith+1 spin.
    '''
    spinsiteslist = list()
    dualbondslist = list()
    #first spin
    i = 0
    j = 2*L - 1
    l = 2
    id_s = ijl_s[(i, j, l)]
    spinsiteslist.append(id_s)
    (ni, nj, nl) = (i, j, l-1)

    allsites = False
    #as long as not every spin site reached: build a new loop
    while (allsites == False):
        loopclosed = False
        #as long as current loop not closed: go to a new site (i, j)
        while(loopclosed == False):
            sitedone = False
            #as long as the spin sites linked to site (i, j) haven't all been reached: nl->nl-1
            while(sitedone == False):
                #update the spins depending on the dimer between them
                id_ns = ijl_s[ni, nj, nl] #
                spinsiteslist.append(id_ns)
                dualbondslist.append(s2_d[id_s, id_ns])
                id_s = id_ns #save new spin site index as new old spin site index
                if (nl > 0):
                    nl = nl-1
                else: # if nl = 0, the next site is ni + 1, nl = 2
                    sitedone = True
            ni = ni + 1
            nl = 2
            (ni, nj, nl) = fixbc(ni, nj, nl, L)
            if ijl_s[(ni, nj, nl)] in spinsiteslist and (ni, nj, nl) == (i, j, l):
                loopclosed = True # when the loop is closed, move to the next one

        id_s = ijl_s[fixbc(i-1, j, 0, L)] # take the new starting point
        i = i
        j = j-1 # id the starting point for the new loop
        l = 2
        (ni, nj, nl) = (i, j, l)
        #check whether this is a spin site which was already visited
        if ijl_s[(i, j, l)] in spinsiteslist:
            allsites = True

    return spinsiteslist, dualbondslist


# In[ ]:

def nsitesconnections(d_ijl, ijl_d, L):
    '''
        For each dual bond, which are the other dual bonds which are touching it through an "n" site
        (in the kagomé case, that's a site with 6 dualbonds)
    '''
    # the dual bond is connected to each dual bond on the same (ij) n site, only not itself: l =/= nl
    d_nd = np.array([[ijl_d[(i,j,nl)] for nl in range(6) if (nl != l)] for (i,j,l) in d_ijl], dtype = 'int32')
    # using that the lists will be ordered in the same way
    # no issue with the boundary conditions    int ndualbonds = -1;
    return d_nd


# In[ ]:

def vsitesconnections(d_ijl, ijl_d, L):
    '''
        For each dual bond, which are the other dual bonds which are touching it through an "v" site
        (in the kagomé case, that's a site with 3 dual bonds)
    '''
    # first, a list for each of the six l values on how to find the neighbours
    # (increase i, increase j, new l)
    nextdualbonds = [[(0, -1, 2), (1, -1, 4)],
              [(1, -1, 3), (1, 0, 5)],
              [(1, 0, 4), (0, 1, 0)],
              [(0, 1, 5), (-1, 1, 1)],
              [(-1, 1, 0), (-1, 0, 2)],
              [(-1, 0, 1), (0, -1, 3)]]
    # this would give the following table, except we have to fix boundary conditions first
    d_vd = [[(i + nextdualbonds[l][u][0], j + nextdualbonds[l][u][1], nextdualbonds[l][u][2]) for u in range(2)] for (i, j, l) in d_ijl]

    # finally, create the list
    d_vd = np.array([[ijl_d[fixbc(ni, nj, nl, L)] for (ni, nj, nl) in dimd] for dimd in d_vd], dtype='int32')
    return d_vd


# In[ ]:

def windingtable(d_ijl, L):
    '''
        For each dual bond, is it on one of the two lines which are used to count the winding numbers?
    '''
    d_wn = np.zeros((len(d_ijl), 2), dtype = 'int32')
    for d, (i, j, l) in enumerate(d_ijl) :
        # First winding number
        if i == 0:
            if j > L-2 and j < 2*L-1:
                if l == 1:
                    d_wn[d,0] = 1
            if j == L - 1:
                if l == 0:
                    d_wn[d,0] = 1 #other case handled above
        if j == 2*L-1:
            if i > 0 and i < L:
                if l == 0:
                    d_wn[d,0] = 1
        if i == 1:
            if j > L-2 and j < 2*L-1:
                if l == 4:
                    d_wn[d,0] = 1
        if j == 2*L-2:
            if i > 0 and i <= L:
                if l == 3:
                    d_wn[d,0] = 1
        #Second winding number
        if i+j == L-1:
            if j != 0:
                if l == 2:
                    d_wn[d,1] = 1
        if i+j == L:
            if j != 0:
                if l == 5:
                    d_wn[d,1] = 1
        if j == 0:
            if i >= L and i <= 2*L-1:
                if l == 3:
                    d_wn[d,1] = 1
            if i == 2*L-1:
                if l == 2:
                    d_wn[d,1] = 1
        if j == 1:
            if i >= L-1 and i < 2*L-1:
                if l == 0:
                    d_wn[d,1] = 1
    return d_wn


# In[ ]:

def d_J2d(d_ijl, ijl_d, L):
    d_J2d = np.array([[[ijl_d[(i, j, nl)]] for nl in [(l-1)%6, (l+1)%6]]
                      for (i, j, l) in d_ijl], dtype = 'int32')
    return d_J2d


# In[ ]:

def d_J3d(d_ijl, ijl_d, L):
    nextj3dualbonds = [[(0, -1, 3), (1, -1, 3)],
                    [(1, -1, 4), (1, 0, 4)],
                    [(1, 0, 5), (0, 1, 5)],
                    [(0, 1, 0), (-1, 1, 0)],
                    [(-1, 1, 1), (-1, 0, 1)],# relative location of dualbonds
                    [(-1, 0, 2), (0, -1, 2)]] # connected via j3 paths
    d_J3d = [[[(i + nextj3dualbonds[l][u][0], j + nextj3dualbonds[l][u][1], 
                nextj3dualbonds[l][u][2])] for u in range (2)] 
             for (i, j, l) in d_ijl]
    # fixing the boundary conditions:
    d_J3d = np.array([[[ijl_d[fixbc(ni, nj, nl, L)] for (ni, nj, nl) in path]
                       for path in dimd_paths] for dimd_paths in d_J3d], 
                     dtype = 'int32') 
    return d_J3d


# In[ ]:



