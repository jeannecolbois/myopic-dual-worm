#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import dimers as dim


# In[ ]:


def createdualtable(L):
    '''
        Creates the table of dual bonds corresponding to a dice lattice of side size L.
        Returns a table identifing an int with the three coordinates of the dual bond 
        and a dictionnary identifying the three coordinates with the dual bond's int 
        index. This allows to handle other relations between dual bonds in an
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
        Creates the table of spin sites corresponding to a dice lattice 
        of side size L.
        Returns a table identifing an int with the three coordinates of 
        the spin site and a dictionnary identifying the
        three coordinates with the spin site's int index. This allows 
        to handle other relations between spin sites in an
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
        For a lattice side size L, this function handles the periodic 
        boundary conditions by returning the corresponding
        value of i, j, l if they match a point which is just outside 
        the borders of the considered cell.
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


def fullfixbc(i,j,l,L, ijl_s):
    '''
        For a lattice side size L, this function handles the periodic 
        boundary conditions by returning the appropriate values
        of i, j, l if they initially corresponded to neighbouring cells
    '''
    listnei = [(0,0),
               (-2, 1), (-1,2), (1,1),
               (2,-1), (1,-2),(-1,-1)]
    
    (si, sj, sl) = (i, j, l)
    for nei in listnei:
        (ni, nj, nl) = (i+nei[0]*L, j+nei[1]*L, l)
        if (ni, nj, nl) in ijl_s:
            (si, sj, sl) = (ni, nj, nl)
            
    if (si, sj, sl) not in ijl_s:
        raise Exception("(si, sj, sl) = ({0},{1},{2}) not in ijl_s".format(si, sj, sl))
        
    return (si, sj, sl)


# In[ ]:


def dualbondspinsitelinks(d_ijl, ijl_s, L):
    '''
        For a lattice with side size L, this function  returns two tables:
        > d_2s: for each dual bond, which are the 2spin sites around it.
        > s2_d: for each pair of spin sites nearest to one another, which 
        is the dual bond between them (dictionary)
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


def createchargesitestable(L):
    '''
        Creates the table of charge sites corresponding to a dice lattice 
        of side size L.
        Returns a table identifing an int with the three coordinates of 
        the charge site and a dictionnary identifying the
        three coordinates with the charge site's int index. This allows 
        to handle other relations between charge sites in an
        easier way.
    '''
    c_ijl = [(i, j, l) for i in range(2*L) for j in range(2*L) for l in range(2) if (i+j > L-2) and (i+j < 3*L-1)]
    # dictionary
    ijl_c = {}
    for c, triplet in enumerate(c_ijl):
        ijl_c[triplet] = c
    return c_ijl, ijl_c


# In[ ]:


def charge2spins(c_ijl, ijl_s, L):
    '''
        Returns the three spin sites associated with each charge site,
        and a sign associated with the way the charge should be computed
    '''
    relspins = [[(0,0,0),(0,0,1),(1,0,2)],[(0,0,1),(0,0,2),(-1,1,0)]]
    # without worrying about periodic BC:
    c2s = [[(ci+relspins[cl][u][0], cj+relspins[cl][u][1], relspins[cl][u][2])
                      for u in range(3)] for (ci,cj,cl) in c_ijl]
    csign = [2*cl -1 for (ci,cj,cl) in c_ijl]
    # fix the periodic boundary conditions
    c2s = [[ijl_s[fixbc(si,sj,sl,L)] for (si,sj,sl) in cspins]
                    for cspins in c2s]
    return c2s, csign


# In[ ]:


def spin2plaquette(ijl_s, s_ijl, s2_d, L):
    '''
        For a lattice with side size L, this function  returns a table giving the
        four dimers surrounding it (which one would have to flip to flip the spin)
        and the four nn spins.
    '''
    nnspinslinks = [[(0,0,1),(1,0,2),(1,-1,1),(1,-1,2)],
               [(0,0,0),(0,0,2),(1,0,2),(-1,1,0)],
               [(0,0,1),(-1,0,0),(-1,0,1),(-1,1,0)]]
    #without worrying about the PBC:
    nnspins = [[(i+nnspinslinks[l][u][0], j+nnspinslinks[l][u][1],nnspinslinks[l][u][2]) for u in range(4)]
               for (i,j,l) in s_ijl]
    nnspins = [[ijl_s[fixbc(si, sj, sl, L)] for (si,sj,sl) in spinsneighs]
                        for spinsneighs in nnspins]
    s2p = [[s2_d[(s1,s2)] for s2 in spinsneighs] for (s1, spinsneighs) in enumerate(nnspins)]
    nnspins = np.array(nnspins, dtype = 'int32')
    s2p = np.array(s2p, dtype = 'int32')
    
    return nnspins, s2p


# In[ ]:


def path_for_measupdate(s_ijl, ijl_s, s2_d, L, version = 0):
    '''
        Returns a list of spin site indices to update the state
        based on a tip magnetic field.
    '''
    path = []
        
    if version == 0:
        for j in range(0, 2*L): # simple scan
            for twice in range(2):
                for i in range(max(0,L-1-j),min(2*L, 3*L-1-j)):
                    if twice == 0:
                        path.append(ijl_s[(i,j,0)])
                    else:
                        path.append(ijl_s[(i,j,2)])
                        path.append(ijl_s[(i,j,1)])
    elif version == 1: # scan following kagome
        for j in range(0, 2*L):
            for i in range(max(0,L-1-j),min(2*L, 3*L-1-j)):
                for l in range(2,-1,-1):
                    path.append(ijl_s[(i,j,l)])
    elif version == 2: # scan following 2nd neighbour axis
        ## not implemented
        assert False, "Version 2 is not yet implemented"
    elif version == 3: # VERSION 0 WITH COMING BACK
        order = [True, False]
        for j in range(0, 2*L):
            for twice in range(2):
                for forward in order:
                    if forward:
                        for i in range(max(0,L-1-j),min(2*L, 3*L-1-j)):
                            if twice == 0:
                                path.append(ijl_s[(i,j,0)])
                            else:
                                path.append(ijl_s[(i,j,2)])
                                path.append(ijl_s[(i,j,1)])
                    elif not forward:
                        for i in range(min(2*L, 3*L-1-j)-1,max(0,L-1-j)-1,-1):
                            if twice == 0:
                                path.append(ijl_s[(i,j,0)])
                            else:
                                path.append(ijl_s[(i,j,1)])
                                path.append(ijl_s[(i,j,2)])
    elif version == 4: # VERSION 1 WITH COMING BACK
        order = [True, False]
        for j in range(0, 2*L):
            for forward in order:
                if forward:
                    for i in range(max(0,L-1-j),min(2*L, 3*L-1-j)):
                        for l in range(2,-1,-1):
                            path.append(ijl_s[(i,j,l)])
                elif not forward:
                    for i in range(min(2*L, 3*L-1-j)-1,max(0,L-1-j)-1,-1):
                        for l in range(0,3,1):
                            path.append(ijl_s[(i,j,l)])

                    
    elif version == 5: # some variant on version 1 (PBC)
        path, bonus = spins_dimers_for_update(s_ijl, ijl_s, s2_d, L)
    elif version == 6: #some variant on version 0 (PBC)
        for i in range(0,2*L):
            for even in range(2):
                for j in range(max(0, L-1-i), min(2*L, 3*L-1-i)):
                    if even == 0:
                        path.append(ijl_s[(i,j,2)])
                        path.append(ijl_s[fixbc(i-1,j+1,0,L)])
                    else:
                        path.append(ijl_s[(i,j,1)])
    return np.array(path, dtype='int32')


# In[ ]:


def spins_dimers_for_update(s_ijl, ijl_s, s2_d, L):
    '''
        Returns a list of spin site indices and a list of dual bond indices. 
        Going through the spins list allows to map the whole
        spin state of the system. The ith dimer lies between the ith and 
        ith+1 spin.
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
        For each dual bond, which are the other dual bonds which are touching 
        it through an "n" site
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
        For each dual bond, which are the other dual bonds which are touching 
        it through an "v" site
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
        For each dual bond, is it on one of the two lines which are used to 
        count the winding numbers?
    '''
    d_wn = np.zeros((len(d_ijl), 2), dtype = 'int8')
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


################## NEIGHBOURS STRUCTURE #################
def NNpairs(ijl_s, s_ijl, L):
    nnpairslist = [[(0,0,0),(0,0,1)],[(0,0,0),(1,0,2)],[(0,0,0),(1,-1,1)],[(0,0,0),(1,-1,2)],
               [(0,0,1),(1,0,2)],[(1,-1,1),(1,-1,2)]]

    #without worrying about the PBC:
    nnpairs = [[[(i+nnpairslist[p][u][0], j+nnpairslist[p][u][1],nnpairslist[p][u][2]) for u in range(2)]
               for p in range(6)] for (i,j,l) in s_ijl if l == 0]
    nnpairs = [[ijl_s[fixbc(si, sj, sl, L)] for (si, sj, sl) in p] for listsp in nnpairs for p in listsp ] 
    return nnpairs


# In[ ]:


def NN2pairs(ijl_s, s_ijl, L):
    nnpairslist = [[(0,0,0),(1,0,1)],[(0,0,0),(0,0,2)],[(0,0,1),(1,-1,2)],[(0,0,1),(0,1,0)],
                   [(1,0,2),(1,-1,1)],[(1,0,2),(-1,1,0)]]

    #without worrying about the PBC:
    nn2pairs = [[[(i+nnpairslist[p][u][0], j+nnpairslist[p][u][1], nnpairslist[p][u][2]) for u in range(2)]
               for p in range(6)] for (i,j,l) in s_ijl if l == 0]
    nn2pairs = [[ijl_s[fixbc(si, sj, sl, L)] for (si, sj, sl) in p]  for listsp in nn2pairs for p in listsp]
                       
    return nn2pairs


# In[ ]:


def NN3parpairs(ijl_s, s_ijl, L):
    '''
        For later use, this is NN3par
    '''
    nnpairslist = [[(0,0,0),(0,1,0)],[(0,0,0),(-1,1,0)],[(0,0,1),(1,-1,1)],[(0,0,1),(-1,0,1)],
                   [(0,0,2),(1,0,2)],[(0,0,2),(0,1,2)]]

    #without worrying about the PBC:
    nn3parpairs = [[[(i+nnpairslist[p][u][0], j+nnpairslist[p][u][1], nnpairslist[p][u][2]) for u in range(2)]
               for p in range(6)] for (i,j,l) in s_ijl if l == 0]
    nn3parpairs = [[ijl_s[fixbc(si, sj, sl, L)] for (si, sj, sl) in p]  for listsp in nn3parpairs for p in listsp]
                       
    return nn3parpairs


# In[ ]:


def NN3starpairs(ijl_s, s_ijl, L):
    '''
        For later use, this is NN3star
    '''
    nnpairslist = [[(0,0,0),(-1,0,0)],[(0,0,1),(0,-1,1)],[(0,0,2),(1,-1,2)]]

    #without worrying about the PBC:
    nn3starpairs = [[[(i+nnpairslist[p][u][0], j+nnpairslist[p][u][1], nnpairslist[p][u][2]) for u in range(2)]
               for p in range(3)] for (i,j,l) in s_ijl if l == 0]
    nn3starpairs = [[ijl_s[fixbc(si, sj, sl, L)] for (si, sj, sl) in p]  for listsp in nn3starpairs for p in listsp]

    return nn3starpairs


# In[ ]:


################### ENERGY ##############################


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


def d_J3std(d_ijl, ijl_d, L):
    d_J3std = np.array([[[ijl_d[(i, j, nl)]  for nl in [(nc-1)%6, nc, (nc+1)%6]
                           if nl != l] for nc in [(l-1)%6, l, (l+1)%6]]
                         for (i, j, l) in d_ijl], dtype = 'int32')
    
    return d_J3std


# In[ ]:


def d_J4d(d_ijl, ijl_d, L):
    #list of the surrounding centers (i', j') in the order Left, Bottom Left, Bottom Right, Right
    centers = [[(-1, 0), (0, -1), (1, -1), (1, 0)],
              [(0, -1), (1, -1), (1, 0),(0, 1)],
              [(1, -1), (1, 0), (0, 1), (-1, 1)],
              [(1, 0), (0, 1), (-1, 1), (-1, 0)],
              [(0, 1), (-1, 1), (-1, 0), (0, -1)],
              [(-1, 1), (-1, 0), (0, -1), (1, -1)]]

    #table without fixed bc:
    d_J4d = [[[(i + centers[l][1][0], j + centers[l][1][1], (l+3)%6), 
               (i + centers[l][1][0], j + centers[l][1][1], (l+4)%6)],
              [(i + centers[l][2][0], j + centers[l][2][1], (l+3)%6), 
               (i + centers[l][2][0], j + centers[l][2][1], (l+2)%6)],
              [(i, j, (l+1)%6), (i + centers[l][1][0], 
                                 j + centers[l][1][1], (l+3)%6)],
              [(i, j, (l-1)%6), (i + centers[l][2][0], 
                                 j + centers[l][2][1], (l-3)%6)],
              [(i, j, (l+1)%6), (i + centers[l][3][0], 
                                 j + centers[l][3][1], (l-2)%6)],
              [(i, j, (l-1)%6), (i + centers[l][0][0], 
                                 j + centers[l][0][1], (l+2)%6)]] for (i, j, l) in d_ijl]

    #fix the boundary conditions
    d_J4d = np.array([[[ijl_d[fixbc(ni, nj, nl, L)] for (ni, nj, nl) in path]
                       for path in dimd_paths] for dimd_paths in d_J4d], 
                     dtype = 'int32')
    return d_J4d


# In[ ]:


###### DISTANCES #####


# In[ ]:


def pairseparation(s1, s2, s_pos, n1, n2, Leff, distmax):
    '''
        Given two spin sites s1 and s2, this function returns the *minimal distance*
        between the two sites (considering pbc) and tells if it is less than Leff/2
        and distmax
    '''
    #What we will do is just list all the six possible positions for the spin s1 and take the minimal distance
    pos1 = s_pos[s1]
    pos2 = s_pos[s2]
    pos2list = []
    listnei = [(0, 0), (0, 1), (1, 0), (-1, 1), (-1, 0), (0, -1),(1, -1)]

    for nei in listnei:
        pos2list.append(pos2 + nei[0]*Leff*n1 + nei[1]*Leff*n2)

    distmin = 10*Leff
    lessthanmax = False
    for pos in pos2list:
        dist = np.linalg.norm(pos1 - pos)
        if dist < distmin:
            distmin = dist

    if distmin < min(Leff/2, distmax):
        lessthanmax = True

    return lessthanmax, distmin


# In[ ]:


def sitepairslist(srefs, s_pos, n1, n2, Leff, distmax):
    '''
        For a given structure, this function returns a table containing,
        for each pair (coord s1, coord s2) at distance less than Leff/2, 
        the corresponding distance R and the *indices* s1 and s2 of the 
        spins in positions these positions. 
        We only consider couples containing spins in srefs.
        It returns as well an ordered list of the distances
        and a dictionary associating each distance to a set of spins.
    '''
    
     # for each distance, we get the various spins that are at this distance from a given spin index

    pairs = []
    distmin = Leff
    
   
    for s1 in srefs:
        for s2 in range(len(s_pos)):
            (consider, dist) = pairseparation(s1, s2, s_pos, n1, n2, Leff, distmax)
            if consider:
                if dist < distmin:
                    distmin = dist
                
                pairs.append(((s1, s2), dist))
                
    distances = []
    distances_spins = {}
    for (spair, dist) in pairs:
        dist = np.round(dist, 4)
        if dist != 0:
            if dist in distances:
                distances_spins[dist].append(spair)
            else:
                distances.append(dist)
                distances_spins[dist] = [spair]

    return pairs, sorted(distances), distances_spins


# In[ ]:


def reducedgraphkag(L, s_ijl, ijl_s):
    '''
        For the kagome lattice:
        returns only one position for each spin (i,j,l) location
    '''
    #position
    s_pos = {} #empty dictionary
    ijl_pos = {}
    for s, (i,j,l) in enumerate(s_ijl):
        x = (2*i + j)
        y = j * np.sqrt(3)
        if l == 0:
            x += 1
        if l == 1:
            x += 0.5
            y +=np.sqrt(3) / 2.0
        if l == 2:
            x -= 0.5
            y += np.sqrt(3) / 2.0
        s_pos[s] = np.array((x,y))
        ijl_pos[s_ijl[s]] = np.array((x,y))
    return s_pos, ijl_pos


# In[ ]:


def superlattice(L):
    n1 = np.array([np.sqrt(3)/2, -1/2])
    n2 = np.array([np.sqrt(3)/2, 1/2])
    Leff = 2*np.sqrt(3)*L
    S = np.sqrt(3)/2 * Leff**2

    return n1, n2, Leff, S


# In[ ]:


#def referenceSpins(L, ijl_s):
#    '''
#        Returns the basic unit cell
#    '''
#    srefs = [ijl_s[(L,L,0)], ijl_s[(L,L,1)], ijl_s[(L,L,2)]]
#    return srefs


# In[ ]:


def NearestNeighboursLists(L, distmax, srefs):
    '''
        Returns a list of distances between sites (smaller than distmax) with respect to the 3 reference sites, a dictionary of pairs of sites at a given distance and a list of the nearest neighbour pairs associated with a given site and distance.
    '''
    #dimer table and dictionary:
    (d_ijl, ijl_d) = createdualtable(L)
    #spin table and dictionary
    (s_ijl, ijl_s) = createspinsitetable(L)
    #two spins surrounding each dimer
    (d_2s, s2_d) = dualbondspinsitelinks(d_ijl, ijl_s, L)
    #dimer-dimer connection through entry sites
    d_nd = nsitesconnections(d_ijl, ijl_d, L)
    #dimer-dimer connection through vertex sites
    d_vd = vsitesconnections(d_ijl, ijl_d, L)
    #for each dimer, is it takeing into account in winding number 1 or 2?
    d_wn = windingtable(d_ijl, L)
    #list of spin indices and dimer indices for the loop allowing to update the spin state
    (sidlist, didlist) = spins_dimers_for_update(s_ijl, ijl_s, s2_d, L)
    
    
    #graph
    (s_pos, ijl_pos) = reducedgraph(L, s_ijl, ijl_s)
    pos = list(s_pos.values())
    pos = [list(np.round(posval, 4)) for posval in pos]
    #initialise the superlattice
    (n1, n2, Leff, S) = superlattice(L)
    
    # getting the list of pairs that we're interested in, 
    #srefs = [ijl_s[(L,L,0)], ijl_s[(L,L,1)], ijl_s[(L,L,2)]]
    pairs, distances, distances_spins = sitepairslist(srefs, s_pos, n1,n2,Leff,distmax+0.01)
    
    NNList = [[[] for i in range(len(distances))] for j in range(len(srefs))]
    
    for i in range(len(distances)):
        for pair in distances_spins[distances[i]]:
            for j in range(len(srefs)):
                if srefs[j] in pair:
                    NNList[j][i].append(pair)

    # correct the neighbour lists elements that can cause trouble
    if distmax >= 2:
        distances.insert(3, distances[2])
    if distmax >= 3.5:
        distances.insert(7, distances[6])
    if distmax >= 2:
        for j in range(len(srefs)):
            NNList3_0 = []
            NNList3_1 = []
            for (s1,s2) in NNList[j][2]:
                halfway = np.round((s_pos[s1] + s_pos[s2])/2, 4)
                if list(halfway) in pos:
                    NNList3_0.append((s1,s2))
                else:
                    NNList3_1.append((s1,s2))

            if distmax >= 3.5:
                NNList6_0 = []
                NNList6_1 = []
                for (s1,s2) in NNList[j][5]:
                    halfway = np.round((s_pos[s1] + s_pos[s2])/2, 4)
                    if list(halfway) in pos:
                        NNList6_0.append((s1,s2))
                    else:
                        NNList6_1.append((s1,s2))

                # replacing in NNList
            NNList[j][2] =  NNList3_0
            NNList[j].insert(3, NNList3_1)
            if distmax >= 3.5:
                NNList[j][6] = NNList6_0
                NNList[j].insert(7, NNList6_1)

    return distances, distances_spins, NNList, s_pos


# In[ ]:


def reducedgraph(L, s_ijl, ijl_s):
    '''
        For the kagome lattice:
        returns only one position for each spin (i,j,l) location
    '''
    #position
    s_pos = {} #empty dictionary
    ijl_pos = {}
    for s, (i,j,l) in enumerate(s_ijl):
        x = (2*i + j)
        y = j * np.sqrt(3)
        if l == 0:
            x += 1
        if l == 1:
            x += 0.5
            y +=np.sqrt(3) / 2.0
        if l == 2:
            x -= 0.5
            y += np.sqrt(3) / 2.0
        s_pos[s] = np.array((x,y))
        ijl_pos[s_ijl[s]] = np.array((x,y))
    return s_pos, ijl_pos

