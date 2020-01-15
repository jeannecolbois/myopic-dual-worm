
# coding: utf-8

# In[ ]:


import numpy as np
import dimers as dim
import KagomeFunctions as lattice
from scipy.special import erfc
from time import time
import itertools


# In[ ]:


def NearestNeighboursList(L,distmax):
    '''
        Returns a list of distances between sites (smaller than distmax) with respect to the 3 reference sites, a dictionary of pairs of sites at a given distance and a list of the nearest neighbour pairs associated with a given site and distance.
    '''
    return lattice.NearestNeighboursList(L, distmax)


# In[ ]:


def createdualtable(L):
    '''
        Creates the table of dual bonds corresponding to the dual lattice 
        of side size L.
        Returns a table identifing an int with the three coordinates of 
        the dual bond and a dictionnary identifying the
        three coordinates with the dual bond's int index. This allows to 
        handle other relations between dual bonds in an easier way.
        > d_ijl: table dimer -> coordinates
        > ijl_d: dictionnary coordinates -> dimer
    '''
    return lattice.createdualtable(L)


# In[ ]:


def createspinsitetable(L):
    '''
        Creates the table of spin sites corresponding to a real space 
        lattice with dual of site size L.
        Returns a table identifing an int with the coordinates of the 
        spin site and a dictionnary identifying the
        three coordinates with the spin site's int index. This allows 
        to handle other relations between spin sites in an easier way.
        > d_ijl: table spin -> coordinates
        > ijl_d: dictionnary coordinates -> spins
    '''
    return lattice.creatspinsitetable(L)


# In[ ]:


def dualbondspinsitelinks(d_ijl, ijl_s, L):
    '''
        For a lattice with side size L, this function  returns two tables:
        > d_2s: for each dual bond, which are the 2spin sites around it.
        > s2_d: for each pair of spin sites nearest to one another, which 
        is the dual bond between them (dictionary)
    '''
    return lattice.dualbondspinsitelinks(d_ijl, ijl_s, L)


# In[ ]:


def spin2plaquette(ijl_s, s_ijl, s2_d,L):
    '''
        For a lattice with side size L, this function  returns a table giving the
        four dimers surrounding it (which one would have to flip to flip the spin)
        and the four nn spins.
    '''
    return lattice.spin2plaquette(ijl_s, s_ijl, s2_d,L)


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
    return lattice.createchargesitestable(L)


# In[ ]:


def charge2spins(c_ijl, ijl_s, L):
    '''
        Returns the three spin sites associated with each charge site,
        and a sign associated with the way the charge should be computed
    '''
    return lattice.charge2spins(c_ijl, ijl_s, L)


# In[ ]:


def spins_dimers_for_update(s_ijl, ijl_s, s2_d, L):
    '''
        Returns a list of spin site indices and a list of dual bond indices. 
        Going through the spins list allows to map the whole
        spin state of the system. The ith dimer lies between the ith and ith+1 
        spin.
        > spinsiteslist: list of spin sites in the update order
        > dualbondslist: list of dual bonds in the update order
    '''
    return lattice.spins_dimers_for_update(s_ijl, ijl_s, s2_d, L)


# In[ ]:


def nsitesconnections(d_ijl, ijl_d, L):
    '''
        For each dual bond, which are the other dual bonds which are 
        touching it through an "n" site (in the kagomé case, that's a 
        site with 6 dualbonds)
        > d_nd: array[d] = list of dual bonds connected to d by a n-site
    '''
    return lattice.nsitesconnections(d_ijl, ijl_d, L)


# In[ ]:


def vsitesconnections(d_ijl, ijl_d, L):
    '''
        For each dual bond, which are the other dual bonds which are 
        touching it through an "v" site (in the kagomé case, that's a 
        site with 3 dual bonds)
        > d_vd: array[d] = list of dual bonds connected to d by a v-site
    '''
    return lattice.vsitesconnections(d_ijl, ijl_d, L)


# In[ ]:


def windingtable(d_ijl, L):
    '''
        For each dual bond, is it on one of the two lines which are used 
        to count the winding numbers?
        > d_wn: array[d] = [w1? w2?]
    '''
    return lattice.windingtable(d_ijl, L)


# In[ ]:


def latticeinit(L):
    #dual bond table and dictionary:
    (d_ijl, ijl_d) = lattice.createdualtable(L)
    #spin site table and dictionary
    (s_ijl, ijl_s) = lattice.createspinsitetable(L)
    #two spin sites surrounding each dual bond
    (d_2s, s2_d) = lattice.dualbondspinsitelinks(d_ijl, ijl_s, L)
    #dual bond - dual bond connection through entry sites
    d_nd = lattice.nsitesconnections(d_ijl, ijl_d,L)
    #dual bond - dual bond connection through vertex sites
    d_vd = lattice.vsitesconnections(d_ijl, ijl_d, L)
    #for each dual bond, is it taking into account in winding number 1 or 2?
    d_wn = lattice.windingtable(d_ijl, L)
    #list of spin site indices and dual bond indices for the loop allowing to update the spin state
    (sidlist, didlist) = lattice.spins_dimers_for_update(s_ijl, ijl_s, s2_d, L)

    #charges
    (c_ijl, ijl_c) = lattice.createchargesitestable(L)
    (c2s, csign) =lattice.charge2spins(c_ijl, ijl_s, L)
    
    return d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, d_nd, d_vd, d_wn, sidlist, didlist, c_ijl, ijl_c, c2s, csign


# In[ ]:


def Hamiltonian(couplings, d_ijl, ijl_d, L):
    '''
        Hamitlonian returns a list of couples (coupling value, interaction 
        table) where the interaction table states for each dual bond which 
        are the dual bonds intereacting with it with the mentionned coupling.
        < couplings : dict. for each coupling ('J1', 'J2', 'J3', 'J3st', ...), associates the
        value (use like a matlab struct)
        > hamiltonian = [J1, (J2, d_J2d), (J3, d_J3d), ...]
    '''
    hamiltonian = [couplings['J1']]
    if 'J2' in couplings: # checks if key in dict.keys() but more efficiently (hash)
        J2 = couplings['J2'];
        d_J2d = lattice.d_J2d(d_ijl, ijl_d, L)
        hamiltonian.append((J2, d_J2d))

    if 'J3' in couplings:
        J3 = couplings['J3'];
        d_J3d = lattice.d_J3d(d_ijl, ijl_d, L)
        hamiltonian.append((J3,d_J3d))
            
    if 'J3st' in couplings:
        J3st = couplings['J3st'];
        J3st = J3st/2.0 # we are going to write two paths going each way!
        d_J3std = lattice.d_J3std(d_ijl, ijl_d, L)
        hamiltonian.append((J3st,d_J3std))
    
    if 'J4' in couplings:
        J4 = couplings['J4'];
        d_J4d = lattice.d_J4d(d_ijl, ijl_d, L)
        hamiltonian.append((J4,d_J4d))
        
    return hamiltonian


# In[ ]:


def compute_energy(hamiltonian, state, latsize = 1):
    '''
        Computes the energy of the state state given the hamiltonian 
        (from dualwormfunctions.Hamiltonian) and the lattice size 
        (number of sites)
    '''
    return dim.hamiltonian(hamiltonian, state)/latsize


# In[ ]:


def GivenNeiNrj(state, power, D, J1supp, nei, distances, distances_spins):
    nrj = 0
    dist = distances[nei]
    for (s1, s2) in distances_spins[dist]:
        nrj += D/(dist**power)*state[s1]*state[s2]
        if np.abs(dist-1) < 1e-2:
            nrj += J1supp*D*state[s1]*state[s2]

    return nrj


# In[ ]:


def NeiFromNeiToNrj(state, power, D, J1supp, neimin, neimax, distances, 
                    distances_spins):
    nrj = 0
    for n in range(neimin, neimax+1, 1):
         nrj += GivenNeiNrj(state, power, D, J1supp, n, distances, distances_spins)
    return nrj


# In[ ]:


def FiniteDistNrj(state, power, D, J1supp, neimax, distances, distances_spins):
    '''
        Function computing the energy of the dipolar system up to neimax 
        neighbours (neimax included)
    '''

    nrj = NeiFromNeiToNrj(state, power, D, J1supp, 1, neimax, distances, 
                          distances_spins)

    return nrj


# In[ ]:


def TruncatedNrj(state, n1, n2, Leff, power, s_pos, D, J1supp):
    '''
        Function computing the energy of the dipolar system by truncating 
        in such a way that a spin doesn't interact with its images
    '''

    nrj = 0
    for s1 in range(len(s_pos)):
        for s2 in range(s1, len(s_pos), 1):
            (consider, dist) = lattice.pairseparation(s1, s2, s_pos, n1, 
                                                      n2, Leff,Leff)
            if consider and dist != 0:
                nrj += D/(dist**power)*state[s1]*state[s2]
            if np.abs(dist-1) < 1e-2:
                nrj += J1supp*D*state[s1]*state[s2]

    return nrj


# In[ ]:


def EwaldSum(state, pairslist, s_pos, klat, D, alpha, S, J1supp):
    '''
        Function computing the energy of the dipolar system using the Ewald 
        summation technique.
        /!\ alpha, klat and pairslist must have been chosen consistently or
        the result might be wrong
    '''
    constterm = - 2*D*alpha**3/(3*np.sqrt(pi))*len(state)

    print('constterm ok')
    realspaceterm = 0
    NNenergy = 0
    for (s1, s2), (r1, r2), dist in pairslist:
        if dist != 0:
            term1 = 2*alpha*np.exp(- alpha**2 * dist**2)/(np.sqrt(pi)*dist**2)
            term2 = erfc(alpha*dist)/(dist**3)
            realspaceterm += D*state[s1]*state[s2]*(term1 + term2)
        if np.abs(dist-1) < 1e-5:
            NNenergy += J1supp*D*state[s1]*state[s2]

    print('realspace ok')
    fourierspaceterm = 0
    for kvec in klat:
        k = np.linalg.norm(kvec)

        factor = (2*alpha/np.sqrt(pi) * np.exp(-k**2 / (4 * alpha**2)) 
                  - k*erfc(k/(2*alpha)))
        ijabsum = 0
        for s1 in range(len(s_pos)):
            r1 = s_pos[s1]
            for s2 in range(len(s_pos)):
                r2 = s_pos[s2]
                ijabsum += state[s1] * state[s2] * np.exp( 1j * 
                                                          np.dot(kvec,(r2-r1)) )
        #for (s1, s2), (r1, r2), dist in pairslist:
        #    ijabsum += state[s1] * state[s2] * np.exp( 1j * np.dot(kvec,(r2-r1) ) )

        fourierspaceterm += factor*ijabsum

    print('fourier space ok')

    fourierspaceterm = (pi * D / S) * fourierspaceterm
    E = constterm + realspaceterm + fourierspaceterm + NNenergy
    issue = True
    if np.abs(np.imag(E)/np.real(E)) < 1e-16:
        issue = False

    return issue, np.real(E)


# In[ ]:


############### Neighbour pairs #####################


# In[ ]:


def NNpairs(ijl_s, s_ijl, L):
    return lattice.NNpairs(ijl_s, s_ijl, L)


# In[ ]:


def NN2pairs(ijl_s, s_ijl, L):
    return lattice.NN2pairs(ijl_s, s_ijl, L)


# In[ ]:


def NN3pairs(ijl_s, s_ijl, L):
    return lattice.NN3pairs(ijl_s, s_ijl, L)


# In[ ]:


def NN4pairs(ijl_s, s_ijl, L):
    return lattice.NN4pairs(ijl_s, s_ijl, L)


# In[ ]:


def reducedgraph(L, s_ijl, ijl_s):
    '''
        Returns exactly one position per spin coordinate.
    '''
    return lattice.reducedgraph(L, s_ijl, ijl_s)


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
    
     # for each distance, we get the various spins that are at this distance
        #  from a given spin index

    pairs = []
    distmin = Leff
    
   
    for s1 in srefs:
        for s2 in range(len(s_pos)):
            (consider, dist) = lattice.pairseparation(s1, s2, s_pos, n1, n2,
                                                      Leff, distmax)
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


def dist_sitepairs(s_pos,  n1, n2, Leff):
    '''
        Using sitepairslist, this function returns a list of (sorted) distances and 
        a dictionnary associating each distance with a spin pair.
        < s_pos: spins-> position
        < n1, n2, Leff: output of superlattice, that is, the structure of the PBC.
        > sorted(distances), distances_spins
    '''
    pairs = lattice.sitepairslist(s_pos, n1, n2, Leff)
    distances = []
    distances_spins = {}
    for (spair, spospair, dist) in pairs:
        dist = np.round(dist, 4)
        if dist in distances:
            distances_spins[dist].append(spair)
        else:
            distances.append(dist)
            distances_spins[dist] = [spair]

    return sorted(distances), distances_spins


# In[ ]:


def NeighboursList(L, distmax):
    '''
        Returns a list of distances between sites (smaller than distmax)
        with respect to the lattice reference sites (e.g. 3 for kagome),
        a dictionary of pairs of sites at a given distance and a list of
        the neighbours associated with a given site and distance.
    '''
    #dimer table and dictionary:
    (d_ijl, ijl_d) = lattice.createdualtable(L)
    #spin table and dictionary
    (s_ijl, ijl_s) = lattice.createspinsitetable(L)
    #two spins surrounding each dimer
    (d_2s, s2_d) = lattice.dualbondspinsitelinks(d_ijl, ijl_s, L)
    #dimer-dimer connection through entry sites
    d_nd = lattice.nsitesconnections(d_ijl, ijl_d)
    #dimer-dimer connection through vertex sites
    d_vd = lattice.vsitesconnections(d_ijl, ijl_d, L)
    #for each dimer, is it takeing into account in winding number 1 or 2?
    d_wn = lattice.windingtable(d_ijl, L)
    #list of spin indices and dimer indices for the loop allowing to 
    # update the spin state
    (sidlist, didlist) = lattice.spins_dimers_for_update(s_ijl, ijl_s, 
                                                         s2_d, L)
    
    (s_pos, ijl_pos) = lattice.reducedgraph(L, s_ijl, ijl_s)
    pos = list(s_pos.values())
    pos = [list(np.round(posval, 4)) for posval in pos]
    
    #initialise the superlattice
    (n1, n2, Leff, S) = lattice.superlattice(L)
    
    # getting the list of pairs that we're interested in, 
    srefs = lattice.referenceSpins(L, ijl_s)
    pairs, distances, distances_spins = sitepairslist(srefs, s_pos, n1,
                                                      n2,Leff,distmax)
    
    NNList = [[[] for i in range(len(distances))] for j in range(len(srefs))]
    
    for i in range(len(distances)):
        for pair in distances_spins[distances[i]]:
            for j in range(len(srefs)):
                if srefs[j] in pair:
                    NNList[j][i].append(pair)


    NNList = lattice.inequivalentSites()                
    return distances, distances_spins, NNList, s_pos, srefs
    


# In[ ]:


############## STATES INIT ##############################3


# In[ ]:


def create_temperatures(nt_list, t_list):
    assert(len(t_list) == len(nt_list) + 1)
    nt = 0
    for nte in nt_list:
        nt += nte

    temp_states = np.zeros(nt)

    nt_start = 0
    for id_nt, nte in enumerate(nt_list):
        temp_states[nt_start: nte+nt_start] =            np.linspace(t_list[id_nt], t_list[id_nt + 1], nte)
        nt_start += nte

    return np.unique(temp_states)


# In[ ]:


def create_hfields(nh_list, h_list):
    assert(len(h_list) == len(nh_list) + 1)
    nh = 0
    for nhe in nh_list:
        nh += nhe

    hfields = np.zeros(nh)

    nh_start = 0
    for id_nh, nhe in enumerate(nh_list):
        hfields[nh_start: nhe+nh_start] =            np.linspace(h_list[id_nh], h_list[id_nh + 1], nhe)
        nh_start += nhe

    return np.unique(hfields)


# In[ ]:


def walkerstable(betas, nt, hfields, nh):
    #walker2params = np.array(list(itertools.product(temperatures, hfields)))
    #walker2id = np.array(list(itertools.product(list(range(0,nt)), list(range(0,nh)))))
    #id2walker = np.zeros((nt, nh), dtype='int32')
    #
    #for i in range(walker2id.shape[0]):
    #    tid = walker2id[i][0]
    #    hid = walker2id[i][1]
    #    id2walker[tid, hid] = i
        
        
    ids2walker = np.zeros((nt, nh), dtype='int32')
    #paramstable = np.zeros((nt, nh,3))
    walker2params = np.zeros((nt*nh, 2))
    walker2ids = np.zeros((nt*nh,2), dtype='int32')
    for bid, beta in enumerate(betas):
        for hid, h in enumerate(hfields):
            wid = bid*nh+hid # this is ONLY true during initialization
            #paramstable[bid, hid, :] = np.array((beta, h, wid))
            ids2walker[bid, hid] = wid
            walker2params[wid,:] = np.array((beta, h))
            walker2ids[wid,:] = np.array((bid, hid), dtype = 'int32')
            
    return walker2params, walker2ids, ids2walker


# In[ ]:


def create_log_temperatures(nt_list, t_list):
    assert(len(t_list) == len(nt_list) + 1)
    nt = 0
    for nte in nt_list:
        nt += nte

    temp_states = np.zeros(nt)

    ## Here, we have htat nt_list = [number between t0 and t1, 
    # number between t1 and t2, ...]

    nt_start = 0
    for id_nt, nte in enumerate(nt_list):
        temp_states[nt_start: nt_start + nte] = np.logspace(
            np.log10(t_list[id_nt]), np.log10(t_list[id_nt +1 ]), 
            nte, endpoint=True)
        nt_start +=nte
        
    return np.unique(temp_states)


# In[ ]:


def statesinit(nt, d_ijl, d_2s, s_ijl, hamiltonian, same = False):
    '''
        Random initialization of the states table (dimers) and computing 
        the initial energy
    '''
    #initialize the dimers
    states = [np.array([1 for i in range(len(d_ijl))], dtype='int32') 
              for ignored in range(nt)]

    #initialize the spins randomly
    spinstates = [(np.random.randint(0, 2, size=len(s_ijl))*2 - 1)
                  for i in range(nt)]
    
    #initialise the dimer state according to the spin state
    for t in range(nt):
        for id_dim in range(len(d_ijl)):
            [id_s1, id_s2] = d_2s[id_dim]
            s1 = spinstates[t][id_s1]
            s2 = spinstates[t][id_s2]
            if (s1 == s2):
                states[t][id_dim] = 1
            else:
                states[t][id_dim] = -1
    statesen = [compute_energy(hamiltonian, states[t]) 
                 for t in range(nt)] # energy computed via the function in c++
    
    return states, statesen


# In[ ]:


def onestatecheck(spinstate, state, d_2s):
    '''
        This function checks whether the dimer state and the spin 
        state are compatible
    '''
    mistakes = list()
    for id_d, d in enumerate(state):
        [id_s1, id_s2] = d_2s[id_d]
        s1 = spinstate[id_s1]
        s2 = spinstate[id_s2]
        if (s1 == s2 and d == -1):
            mistakes.append((id_d, id_s1, id_s2))
        if (s1 != s2 and d == 1):
            mistakes.append((id_d, id_s1, id_s2))
    return mistakes


# In[ ]:


def statescheck(spinstates, states, d_2s):
    '''
        This function checks whether the dimer stateS and the spin 
        stateS are compatible
    '''
    for spinstate, state in zip(spinstates, states):
        if len(onestatecheck(spinstate, state, d_2s)) != 0:
            return False
    return True


# In[ ]:


def onestate_dimers2spins(sidlist, didlist, states,
                          spinstates, tid, ncores):
    '''
        For a known state of the dual lattice (i.e. for
        each bond, is 
        there or not a dimer), returns the corresponding
        spin state.
    '''
    stat_temps = [nt]
    dim.updatespinstates(states, spinstates,
                         np.array(stat_temps, dtype='int32'),
                         np.array(sidlist, dtype='int32'), 
                         np.array(didlist, dtype='int32'),
                         ncores)
    return


# In[ ]:


def states_dimers2spins(sidlist, didlist, states, spinstates,
                        nt,ncores):
    stat_temps = list(range(nt))
    dim.updatespinstates(states, spinstates,
                         np.array(stat_temps,dtype='int32'),
                         np.array(sidlist, dtype='int32'),
                         np.array(didlist, dtype='int32'),
                         ncores)
    
    return np.array(spinstates, dtype='int32')


# In[ ]:


############ EVOLUTION ############


# In[ ]:


def measupdatespin(tid, sidlist, states, spinstates,nnspins, s2p, p):
    spinstate = spinstates[tid]
    for sid in range(len(spinstate)):
        s = spinstate[sid]
        if s == 1 :
            #if the spin is down, check if we can flip it
            neispinstates = np.array([spinstate[snei] for snei in nnspins[sid]])
            #if it costs no energy:
            if neispinstates.sum() == 0:
                if np.random.random_sample() < p:
                    #flip the spin
                    spinstates[tid][sid] = -1
                    for did in s2p[sid]:
                        # and flip the corresponding dimers
                        states[tid][did] *= -1
            #endif
        #endif
    #endfor


# In[ ]:


def statistics(tid, resid, hid, reshid, bid, states, statesen, statstables,
               spinstates,statsfunctions, sidlist, didlist, L, s_ijl, ijl_s,
               num_in_bin, stlen, magnfuncid, ids2walker, **kwargs):
    '''
        This function updates the statistics in statstables given the states,
        the states energy, the statistical functions, the list of spins and
        dimers for updates,
        the system size and the number of states in a bin
    '''
    # bin index = integer division of the iteration and the number
    # of iterations in a bin
    #   Before doing any measurement, the spinstate must be updated.
    #   But it is not necessary to update the spinstate
    #   if no measurement is performed, since the code runs on dimer
    #   configurations.
    #   Hence, we can feed the statistics threads with only the temperatures
    #   indices for which we are interested in
    #   the statistics.
    
    m = 0
    
    wid = ids2walker[tid, hid]
    for stat_id in range(len(statstables)): #stat_id: index of the statistical
        #function you're currently looking at
        func_per_site = statsfunctions[stat_id](stlen, states[wid],
                                                statesen[tid, hid], 
                                                spinstates[wid],
                                                s_ijl, ijl_s,m=m,
                                                **kwargs)
        # c2s = c2s, csign = csign,nnlists = nnlists, m = m) 
        #evaluation depends on the temperature index
        if stat_id == magnfuncid:
            m = func_per_site
            
        statstables[stat_id][0][resid][reshid][bid] += func_per_site / num_in_bin 
        #storage depends on the result index
        
        statstables[stat_id][1][resid][reshid][bid] += (func_per_site ** 2) / num_in_bin


# In[ ]:


def tempering(nt, statesen, betas, states, spinstates, swaps):
    '''
        This function proposes as a trial exchanging pairs of
        states at neighbouring temperatures, enforcing detailed balance
        in the ensemble of the set of systems at different
        temperatures.
    '''
    for t in range(nt-1, 0, -1):
        #throw a dice
        if (statesen[t] - statesen[t-1]) * (betas[t] - betas[t-1]) > 0: 
            # if bigger than 0 flip for sure
            #states:
            states[[t-1 , t]] = states[[t, t-1]]
            spinstates[[t-1 , t]] = spinstates[[t, t-1]]
            #energy:
            statesen[[t - 1, t]] = statesen[[t, t - 1]]
            swaps[t] += 1
            swaps[t-1] += 1
        elif np.random.uniform() < np.exp((statesen[t] - statesen[t-1]) 
                                          * (betas[t] - betas[t-1])): 
            #else flip with a certain prob
            #states:
            states[[t-1 , t]] = states[[t, t-1]]
            #energy:
            statesen[[t - 1, t]] = statesen[[t, t - 1]]
            spinstates[[t-1 , t]] = spinstates[[t, t-1]]
            swaps[t] += 1
            swaps[t-1] +=1


# In[ ]:


def replicas(it, nt, nh, statesen, betas, hfields, states, spinstates,
             swapst, swapsh,ids2walker, walker2ids, walker2params):
    '''
        Given the number of temperatures and magnetic fields, 
        the states, and the mapping from walker to parameter ids,
        this function proposes a selection of swaps which are accepted
        or rejected based on detailed balance.
        As the single spin flip update is almost useless in some cases,
        this sometimes happens to be the core of the algorithm.
    '''
    
    if it%4 == 0:
        # even t swap
        for hid in range(nh):
            for tid in range(0,nt-1,2):
                swaptemps(tid, hid, statesen, betas, hfields, ids2walker,
                          walker2ids, walker2params, swapst)
            
    elif it%4 == 1:
        # even h swap:
        for tid in range(0, nt):
            for hid in range(0, nh-1, 2):
                swapfields(tid, hid, statesen, betas, hfields, spinstates,
                           ids2walker, walker2ids, walker2params, swapsh)
                
    elif it%4 == 2:
        #odd t swap
        for hid in range(nh):
            for tid in range(1,nt-1,2):
                swaptemps(tid, hid, statesen, betas, hfields, ids2walker,
                          walker2ids, walker2params, swapst)
                
    elif it%4 == 3:
        # odd h swap:
        for tid in range(0, nt):
            for hid in range(1, nh-1, 2):
                swapfields(tid, hid, statesen, betas, hfields, spinstates,
                           ids2walker, walker2ids, walker2params, swapsh)
                


# In[ ]:


def swaptemps(tid, hid, statesen, betas, hfields,
              ids2walker, walker2ids, walker2params, swapst):
    '''
        Offers to swap temperatures and accepts or reject based on
        detailed balance
    '''
    wid = ids2walker[tid, hid]
    wid2 = ids2walker[tid+1, hid]
    
    swap = False
    if ((statesen[tid+1, hid] - statesen[tid, hid])*
        (betas[tid+1] - betas[tid]))> 0:
        swap = True
    elif (np.random.uniform() <           np.exp((statesen[tid+1, hid] - statesen[tid,hid])
                 * (betas[tid+1] - betas[tid]))):
        swap = True
        
    if swap:
        swapst[tid]+=1
        
        ids2walker[tid, hid], ids2walker[tid+1, hid] =        ids2walker[tid+1, hid], ids2walker[tid, hid]
        
        walker2ids[wid, 0], walker2ids[wid2, 0] =        walker2ids[wid2, 0], walker2ids[wid,0] # 0, cause we are swapping the temperatures
        
        
        walker2params[wid, 0], walker2params[wid2,0] =        walker2params[wid2, 0], walker2params[wid,0]
        
        # update the energy
        statesen[tid, hid],  statesen[tid+1, hid] =        statesen[tid+1, hid],  statesen[tid, hid]
        
    


# In[ ]:


def swapfields(tid, hid, statesen, betas, hfields, spinstates,
               ids2walker, walker2ids, walker2params, swapsh):
    '''
        Offers to swap magnetic fields and accepts or reject
        based on detailed balance
    '''
    wid = ids2walker[tid, hid]
    wid2 = ids2walker[tid, hid+1]
    
    swap = False
    if (betas[tid]*(hfields[hid+1] - hfields[hid])
        *(spinstates[wid].sum() - spinstates[wid2].sum()))  > 0:
        swap = True
    elif (np.random.uniform() <          np.exp(betas[tid]*(hfields[hid+1] - hfields[hid])
                 *(spinstates[wid].sum() - spinstates[wid2].sum()))):
        swap = True
        
    if swap:
        swapsh[hid] += 1
        ids2walker[tid, hid], ids2walker[tid, hid+1] =        ids2walker[tid, hid+1], ids2walker[tid, hid]
        
        walker2ids[wid, 1], walker2ids[wid2,1] =        walker2ids[wid2, 1], walker2ids[wid,1] 
        
        walker2params[wid, 1], walker2params[wid2,1] =        walker2params[wid2, 1], walker2params[wid,1]
        
        # update the energy
        deltaE = statesen[tid, hid+1] - statesen[tid, hid]
        
        statesen[tid, hid] += deltaE +        (hfields[hid+1] - hfields[hid])*spinstates[wid2].sum()
        statesen[tid, hid+1] += -deltaE +        (hfields[hid] - hfields[hid+1])*spinstates[wid].sum()


# In[ ]:


def mcs_swaps(states, spinstates, statesen, 
              betas, stat_temps, stat_fields, **kwargs):
    '''
        < keyword arguments:
                'nb' : number of bins
                'num_in_bin' : number of meas in each bin
                'iterworm' : number of worm iterations per sweep
                'nitermax' : limiting the size of worms
                'check' : whether to check or not that the states remained 
                consistent
                ---- thermodynamic
                'statsfunctions':
                'nt':
                'nnlists': lists of nearest neighbours
                ---- tables for loop building (worm algorithm)
                'hamiltonian': see hamiltonian function
                'd_nd': see nsitesconnections function
                'd_vd': see vsitesconnections function
                'd_wn': see windingtable function
                'd_2s': see dualbondspinsitelink function
                's2_d': see dualbondspinsitelink function
                'sidlist': see spin_dimers_for_update function
                'didlist': see spin_dimers_for_update function
                ---- system size
                's_ijl': see s_ijl
                'ijl_s': idem
                'L': system size
                'ncores':ncores
                'hfields':hfields
                'walker2params': walker2params
                'walker2ids': walker index -> parameters indices
        < states, spins states = tables that will be updated as the new 
        states and spinstates get computed
        < statesen : energy of the states
        < betas : list of inverse temperatures of the states
    '''
    ## Parse the keywords arguments
    #bins
    nb = kwargs.get('nb', None)
    num_in_bin = kwargs.get('num_in_bin',None)
    
    #iteration and simulations parameters
    iterworm = kwargs.get('iterworm',None)
    nrps = kwargs.get('nrps')
    nitermax = kwargs.get('nitermax',None)
    ncores = kwargs.get('ncores',4)
    measperiod = kwargs.get('measperiod', 1)

    #spin structure table
    s_ijl = kwargs.get('s_ijl',None)
    stlen = len(s_ijl)
    ijl_s = kwargs.get('ijl_s', None)
    d_nd = kwargs.get('d_nd',None)
    d_vd = kwargs.get('d_vd',None)
    d_wn = kwargs.get('d_wn',None)
    d_2s = kwargs.get('d_2s',None)
    s2_d = kwargs.get('s2_d',None)
    sidlist = kwargs.get('sidlist',None)
    didlist = kwargs.get('didlist',None)
    L = kwargs.get('L', None)
    nnlists = kwargs.get('nnlists',[])
    
    #charges
    c2s = kwargs.get('c2s', None)
    csign = kwargs.get('csign', None)
    
    # check state?
    check = kwargs.get('check', None)
    
    #hamiltonian
    hamiltonian = kwargs.get('hamiltonian',None)
    
    # statistics to measure and how
    statsfunctions = kwargs.get('statsfunctions',[])
    magnfuncid = kwargs.get('magnfuncid', -1)
    measupdate = kwargs.get('measupdate', False)
    p = kwargs.get('p', 1)
    if p == 0:
        measupdate = False
    nnspins = kwargs.get('nnspins',None)
    s2p = kwargs.get('s2p', None)
    nt = kwargs.get('nt',None)
    randspinupdate = kwargs.get('randspinupdate', True)
    
    #structure for replica:
    hfields = kwargs.get('hfields', None)
    nh = kwargs.get('nh',None)
    walker2params = kwargs.get('walker2params',[])
    walker2ids = kwargs.get('walker2ids', [])
    ids2walker = kwargs.get('ids2walker', [])
    ssf = kwargs.get('ssf', False)
    ################################
    # actual code
    ################################

    ## Define the table for statistics
    if len(statsfunctions) != 0:
            statstables = [(np.zeros((len(stat_temps), len(stat_fields), nb)).tolist(),
                           np.zeros((len(stat_temps), len(stat_fields), nb)).tolist())
                           for i in range(len(statsfunctions))]
    else:
        statstables =  []
        
    stat_paramsid = np.array(list(itertools.product(stat_temps, stat_fields)))
    
    ## Iterate
    itermcs = nb*num_in_bin*measperiod
    print("itermcs = ", itermcs)
    print("iterreplicas = ", nrps)
    swapst = np.array([0 for tid in range(nt)], dtype='int32')
    swapsh = np.array([0 for hid in range(nh)], dtype='int32')
    
    failedupdates = np.array([[0 for hid in range(nh)] for bid in range(nt)],dtype ='int32')
    
    t_join = 0
    t_spins = 0
    t_tempering = 0
    t_stat = 0
    
    print("statsfunctions", statsfunctions)


    for it in range(itermcs):
        #### EVOLVE using the mcsevolve function of the dimer
        #### module (C)
        # Note that states, betas, statesen get updated
        if nh == 1 and hfields[0] == 0.0 and not ssf:
            t1 = time()
            dim.mcsevolve(hamiltonian, states, betas, statesen,
                          failedupdates, d_nd, d_vd, d_wn,
                          iterworm, nitermax, ncores)
            t2 = time()
            t_join += (t2-t1)/itermcs
        else:
            if nh == 1 and not ssf:
                t1 = time()
                dim.magneticmcsevolve(hamiltonian, hfields[0],
                                      states, spinstates,
                                      d_nd, d_vd, d_wn, sidlist,
                                      didlist, betas,
                                      statesen, failedupdates,
                                      nitermax, iterworm,
                                      ncores)
                t2 = time()
                t_join += (t2-t1)/itermcs
            else:
                t1 = time()
                dim.ssfsevolve(hamiltonian[0], states, spinstates,
                               np.array(s2p, dtype = 'int32'),
                               walker2params, walker2ids, statesen,
                               failedupdates, ncores,
                               iterworm)
                t2 = time()
                t_join += (t2-t1)/itermcs


        #### TEMPERING perform "parallel" tempering
        if nh == 1:
            print("tempering??")
            tempering(nt, statesen, betas, states, spinstates,
                      swapst)
        else: # replicas in both
            for riter in range(nrps):
                replicas(it, nt, nh, statesen, betas, hfields, states, 
                         spinstates, swapst, swapsh, ids2walker,
                         walker2ids, walker2params)
    
        t3 = time()
        t_tempering +=(t3-t2)/itermcs
        
        #### STATS update the spin states
        if (len(statsfunctions) != 0 or check) and nh == 1 and hfields[0] == 0:
            # update the mapids2walker function
            def mapids2walker(x):
                return ids2walker[x[0],x[1]]
            
            stat_walkers = np.array(list(map(f, stat_paramsid)))
            
            dim.updatespinstates(states, spinstates,
                                 np.array(stat_walkers, dtype='int32'), 
                                 np.array(sidlist, dtype='int32'),
                                 np.array(didlist, dtype='int32'), 
                                 ncores, randspinupdate)
        # if h !=0 the spinstates have been updated already

        if measperiod == 1 or it%measperiod == 0:
            binid = (it//measperiod)//num_in_bin
            if len(statsfunctions) != 0 or check:
                #print(bid)
                if measupdate:                
                    dim.measupdates(states, spinstates,
                                    np.array(stat_temps, dtype='int32'),
                                    np.array(sidlist, dtype = 'int32'),
                                    np.array(didlist, dtype='int32'),
                                    np.array(nnspins,dtype = 'int32'), 
                                    np.array(s2p, dtype = 'int32'), 
                                    ncores, p);


                for resid,tid in enumerate(stat_temps):
                    for reshid, hid in enumerate(stat_fields):
                        statistics(tid, resid, hid, reshid, binid, states, statesen,
                                   statstables,  spinstates,statsfunctions,
                                   sidlist, didlist, L, s_ijl, ijl_s,
                                   num_in_bin, stlen,
                                   magnfuncid, ids2walker,\
                                   c2s = c2s, csign = csign,nnlists = nnlists)
                # it would probably be worth it to parallelise this in c++
                # ideally I should do it before the spins update, then 
                # perform the spin update and possibly the replicas in c++.
            
        t4 = time()
        t_spins += (t4-t3)/itermcs

    # ENDFOR

    # verifications

    if len(statsfunctions) != 0 or check:
        for t in stat_temps:
            assert len(onestatecheck(spinstates[t], states[t], d_2s)) == 0,            'Loss of consistency at temperature index {0}'.format(t)
    ttot = time()

    print('Time for mcsevolve = {0}'.format(t_join))
    print('Time for tempering = {0}'.format(t_tempering))
    print('Time for mapping to spins + computing statistics= {0}'.format(t_spins))
    if ssf:
        failedupdates = failedupdates/len(s_ijl)
    return statstables, swapst, swapsh, failedupdates
    
    

