
# coding: utf-8

# 18.02.2019; Last update 16.12.2019
# Author : Jeanne Colbois
# Please send any comments, questions or remarks to Jeanne Colbois: jeanne.colbois@epfl.ch.
# The author would appreciate to be cited in uses of this code, and would be very happy to hear about potential nice developments.

# Aim: study the AF Ising model up to 3rd nearest neighbours couplings
# Idea: use the Myopic Algorithm (Rakala 16), which allows to build a rejection-free worm update on the dual lattice of Kagome, the dice lattice
# The critical steps of the algorithm are implemented in C++ but the rest is done in python
# All the work is done on the dual lattice for now

# In[1]:


# import
import numpy as np # maths
from functools import lru_cache

# In[2]:


#############################################################
  ## DESCRIBING THE DUAL LATTICE  AND DIMER INTERACTIONS ##
#############################################################


# In[3]:


### DUAL BOND TABLE
# make the mapping between the dual index d \in {0,...,D-1} and the numbering (ijl)
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


# In[4]:


### SPIN SITES TABLE
# make the mapping between the spinsite index s\in {0,..., S-1} and the numbering (ijl)
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

def OBCmapping(L, s_ijl, filename):
    s42_ijl = []
    ijl_s42 = {}
    s42_realpos = []
    s42_pos = []
    pos_s42 ={}
    configlist = np.loadtxt(filename,delimiter=',')
    
    if L == 3:
        s42_ijl = [(0,3,1), (0,2,1), 
                   (0,4,0), (1,3,2), (0,3,0), (1,2,2), (0,2,0), (1,1,2),
                   (1,3,1), (1,2,1), (1,1,1),
                   (1,4,0), (2,3,2), (1,3,0), (2,2,2), (1,2,0), (2,1,2), (1,1,0), (2,0,2),
                   (2,3,1), (2,2,1), (2,1,1), (2,0,1), 
                   (3,3,2), (2,3,0), (3,2,2), (2,2,0), (3,1,2), (2,1,0), (3,0,2), (2,0,0),
                   (3,2,1), (3,1,1), (3,0,1),
                   (4,2,2), (3,2,0), (4,1,2), (3,1,0), (4,0,2), (3,0,0),
                   (4,1,1), (4,0,1)]
        for s42, (i,j,l) in enumerate(s42_ijl):
            ijl_s42[(i,j,l)] = s42
            s42_realpos.append(np.array((configlist[s42][0], configlist[s42][1])))
            x = 2*i + j
            y = j * np.sqrt(3)
            if l == 0:
                x += 1 
            if l == 1:
                x += 1 / 2.0
                y += np.sqrt(3) / 2.0
            if l == 2:
                x -= 1 / 2.0
                y += np.sqrt(3) / 2.0
            s42_pos.append(np.array((x,y)))
    return s42_ijl, ijl_s42, s42_realpos, s42_pos, pos_s42, configlist

def OBC_NthNeighbourList(foldername, N):
    filename = ''
    filename1 = ''
    firstpairs = np.array(())
    secondpairs = np.array(())
    
    if N == 1:
        filename = 'NNTable.csv'
    elif N == 2: 
        filename = 'NNNTable.csv'
    elif N == 3:
        filename = 'ThirdStarTable.csv'
        filename1 = 'ThirdParTable.csv'
    elif N == 4:
        filename = 'FourthTable.csv'
    elif N == 5:
        filename = 'FifthTable.csv'
    elif N == 6:
        filename = 'SixthStarTable.csv'
        filename1 = 'SixthParTable.csv'
    elif N == 7:
        filename = 'SeventhTable.csv'
    elif N == 8:
        filename = 'EigthStarTable.csv'
        filename1 = 'EigthParTable.csv'
    elif N == 9:
        filename = 'NinthTable.csv'
    else:
        filename = 'Invalid N'
    
    if filename != 'Invalid N':
        print(foldername + filename)
        firstpairs = np.loadtxt(foldername+filename, dtype =('i4','i4'), delimiter=',')
        if filename1 != '':
            print(foldername + filename1)
            secondpairs= np.loadtxt(foldername+filename1, dtype =('i4','i4'), delimiter = ',')
    else:
        print('filename')
    
    return firstpairs, secondpairs

def OBC_NeighboursList(foldername):
    listpairs = []
    for N in range(1,10):
        (firstpairs,secondpairs) = OBC_NthNeighbourList(foldername, N)
        listpairs.append((firstpairs,secondpairs))
    return listpairs
# In[5]:

@lru_cache(maxsize = None)
def graphkag(L, a):
    '''
        For the kagomé lattice:
        Returns two vertex <-> (i, j, l) tables, a table linking edge to the two corresponding vertices, as well as a dictionary giving the position (x,y) of each vertex
    '''
    #vertices table
    sv_ijl = [(i, j, l) for i in range(-1, 2*L+1) for j in range(-1, 2*L+1) for l in range(3) if (i+j >= L-2) and (i+j <= 3*L - 1)]
    
    #vertices dictionary:
    ijl_sv = {}
    for sv, triplet in enumerate(sv_ijl):
        ijl_sv[triplet] = sv
    
    #create the edges (sv1, sv2)
    #table of where to look at 
    nv = [[(0, 0, 1), (1, 0, 2)],[(-1, 1, 0), (1, 0, 2)], [(0, 0, 1), (-1, 1, 0)]]
    #edge -> vertex: l from 0 to 5 indicates the edge
    e_2sv = [((i, j, l),(i + nv[l][u][0], j + nv[l][u][1], nv[l][u][2])) for i in range(-1, 2*L+1) for j in range(-1, 2*L+1) for l in range(3) for u in range(2) if (i+j >= L-2) and (i+j <= 3*L - 1)]
    e_2sv = [(ijl_sv[i, j, l], ijl_sv[ni, nj, nl]) for ((i, j, l), (ni, nj, nl)) in e_2sv if (ni, nj, nl) in sv_ijl]
    #position
    pos = {} #empty dictionary
    for sv, (i,j,l) in enumerate(sv_ijl):
        x = a * (i + j / 2.0)
        y = a * j * np.sqrt(3) / 2.0
        if l == 0:
            x += a / 2.0
        if l == 1:
            x += a / 4.0
            y += a * np.sqrt(3) / 4.0
        if l == 2:
            x -= a / 4.0
            y += a * np.sqrt(3) / 4.0
        pos[sv] = (x,y)
    return sv_ijl, ijl_sv, e_2sv, pos


# create a function to fix the periodic boundary conditions
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

def fullfixbc(i,j,l,L, ijl_s, xtrafix = False, numxtrafix = [], check = True):
    '''
        For a lattice side size L, this function handles the periodic 
        boundary conditions by returning the appropriate values
        of i, j, l if they initially corresponded to neighbouring cells
    '''
    listnei = [(0,0),
               (-2, 1), (-1,2), (1,1),
               (2,-1), (1,-2),(-1,-1)]
    
    (si, sj, sl) = (0,0,0)
    # first attempt:
    for nei in listnei:
        (ni, nj, nl) = (i+nei[0]*L, j+nei[1]*L, l)
        if (ni, nj, nl) in ijl_s:
            (si, sj, sl) = (ni, nj, nl)
            
    if (si, sj, sl) not in ijl_s and xtrafix:
        # second attempt:
        notdone = True
        neiid = 0
        while notdone and neiid < len(listnei):
            nei = listnei[neiid]
            (ni, nj, nl) = (i+nei[0]*L, j+nei[1]*L, l)
            (ni, nj, nl) = fullfixbc(ni, nj, nl, L, ijl_s, check = False)
            if (ni, nj, nl) in ijl_s:
                (si, sj, sl) = (ni, nj, nl)
                notdone = False
                numxtrafix[0]+=1
            else:
                neiid += 1
    
    if (si, sj, sl) not in ijl_s and check:
        raise Exception("(si, sj, sl) = ({0},{1},{2}) not in ijl_s (initially ({3},{4},{5})".format(si, sj, sl,
                                                                                                   i, j, l))
        
    return (si, sj, sl)
# In[6]:


### DUAL BONDS - SPIN SITES LINKS
# for each dual bond, two spin sites to which it's related
#list of neighbours for the six l values
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


# In[7]:


#lists of dual bonds and of spin sites to go through in the right order to update the spin state
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


# In[8]:


### DUAL BOND ENTRY CONNECTIONS (n sites, 5 "neighbours")
def nsitesconnections(d_ijl, ijl_d):
    '''
        For each dual bond, which are the other dual bonds which are touching it through an "n" site
        (in the kagomé case, that's a site with 6 dualbonds)
    '''
    # the dual bond is connected to each dual bond on the same (ij) n site, only not itself: l =/= nl
    d_nd = np.array([[ijl_d[(i,j,nl)] for nl in range(6) if (nl != l)] for (i,j,l) in d_ijl], dtype = 'int32')
    # using that the lists will be ordered in the same way
    # no issue with the boundary conditions    int ndualbonds = -1;
    return d_nd


# In[9]:


### DUAL BOND VERTEX CONNECTIONS (v sites, 2 "neighbours")

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

