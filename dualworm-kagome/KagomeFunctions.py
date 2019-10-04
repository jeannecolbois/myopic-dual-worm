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


def TruncatedNrj(state, n1, n2, Leff, power, s_pos, D, J1supp):
    '''
        Function computing the energy of the dipolar system by truncating in such a way that a spin doesn't interact with its images
    '''

    nrj = 0
    for s1 in range(len(s_pos)):
        for s2 in range(s1, len(s_pos), 1):
            (consider, dist) = pairseparation(s1, s2, s_pos, n1, n2, Leff,Leff)
            if consider and dist != 0:
                nrj += D/(dist**power)*state[s1]*state[s2]
            if np.abs(dist-1) < 1e-2:
                nrj += J1supp*D*state[s1]*state[s2]

    return nrj


# In[ ]:


def FiniteDistNrj(state, power, D, J1supp, neimax, distances, distances_spins):
    '''
        Function computing the energy of the dipolar system up to neimax neighbours (neimax included)
    '''

    nrj = NeiFromNeiToNrj(state, power, D, J1supp, 1, neimax, distances, distances_spins)

    return nrj


# In[ ]:


def NeiFromNeiToNrj(state, power, D, J1supp, neimin, neimax, distances, distances_spins):
    nrj = 0
    for n in range(neimin, neimax+1, 1):
         nrj += GivenNeiNrj(state, power, D, J1supp, n, distances, distances_spins)
    return nrj


# In[ ]:


def GivenNeiNrj(state, power, D, J1supp, nei, distances, distances_spins):
    nrj = 0
    dist = distances[nei]
    for (s1, s2) in distances_spins[dist]:
        nrj += D/(dist**power)*state[s1]*state[s2]
        if np.abs(dist-1) < 1e-2:
            nrj += J1supp*D*state[s1]*state[s2]

    return nrj


# In[15]:


from scipy.special import erfc
def EwaldSum(state, pairslist, s_pos, klat, D, alpha, S, J1supp):
    '''
        Function computing the energy of the dipolar system using the Ewald summation technique.
        /!\ alpha, klat and pairslist must have been chosen consistently or the result might be wrong
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

        factor = 2*alpha/np.sqrt(pi) * np.exp(-k**2 / (4 * alpha**2)) - k*erfc(k/(2*alpha))
        ijabsum = 0
        for s1 in range(len(s_pos)):
            r1 = s_pos[s1]
            for s2 in range(len(s_pos)):
                r2 = s_pos[s2]
                ijabsum += state[s1] * state[s2] * np.exp( 1j * np.dot(kvec,(r2-r1)) )
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

# In[26]:



#############################################################
                ## STATES EVOLUTION ##
#############################################################


# In[29]:


### Define a function performing a series of MCS and then a series of swaps
# parallelism will be added later
from time import time
from threading import Thread
def mcs_swaps(nb, num_in_bin, iterworm, saveloops, temp_loops, temp_lenloops, check, statsfunctions, nt, stat_temps, hamiltonian, d_nd, d_vd, d_wn, d_2s, s2_d, sidlist, didlist, L, states, spinstates, en_states, beta_states, s_ijl, nitermax):
    '''
        This function creates the mcs class and then makes itermcs times the following:
            - ignored evolution for ignoredsteps (evolution during which the loops aren't registered)
            - considered evolution for consideredsteps (evolution during which the loops are registered)
            - swaps
    '''
    ############
    class mcs_withloops(Thread): #create a class inheritating from thread
        '''
            This class allows to make each thread evolve independently and save the loops
        '''
        def __init__(self, t, iterworm): #constructor
            Thread.__init__(self) # call to the thread constructor
            self.t = t # temperature index
            self.num = iterworm
            self.updates = list() # list of the updates which are created
            self.updatelengths = list() #list of length of updates

        def run(self):
            #with loops
            for _ in range(self.num):
                #update the state (notice that we work only with the DIMER state in the update code
                result = dim.manydualworms(hamiltonian, states[self.t], d_nd, d_vd, d_wn, beta_states[self.t], saveloops[self.t], nitermax) # the evolve function returns the difference in energy
                #update the energy
                en_states[self.t] += result[0]
                #statistics:
                #update loop
                self.updates.append(result[1])
                self.updatelengths.append(result[2])

    class mcs_withoutloops(Thread): #create a class inheritating from thread
        '''
            This class allows to make each thread evolve independently and not save the loops
        '''
        def __init__(self, t, iterworm): #constructor
            Thread.__init__(self) # call to the thread constructor
            self.t = t # temperature index
            self.num = iterworm
            self.updatelengths = list()

        def run(self):
            #without loops
            for _ in range(self.num):
                #update the state
                result = dim.manydualworms(hamiltonian, states[self.t], d_nd, d_vd, d_wn, beta_states[self.t], False, nitermax) # the evolve function returns the difference in energy
                #update the energy
                en_states[self.t] += result[0]

    class statistics(Thread):
        '''
            This class allows to update each spinstate independently and make the corresponding measurements
        '''
        def __init__(self, t, resid):
            Thread.__init__(self)
            self.t = t #temperature index
            self.resid = resid #index in the result list (eg if you only measure the smallest and highest temperature, resid = 1 for the highest temperature)

        def run(self):
            # Before doing the statistics, the spinstate must be updated
            spinstates[self.t] = onestate_dimers2spins(sidlist, didlist, L, states[self.t])
            # bin index = integer division of the iteration and the number of iterations in a bin
            b = it//num_in_bin
            for stat_id, stat_tuple in enumerate(statstables): #stat_id: index of the statistical function you're currently looking at
                func_per_site = statsfunctions[stat_id](states[self.t], en_states[self.t], spinstates[self.t], s_ijl) #evaluation depends on the temperature index
                stat_tuple[0][self.resid][b] += func_per_site / num_in_bin #storage depends on the result index
                stat_tuple[1][self.resid][b] += (func_per_site ** 2) / num_in_bin
    ##############

    # TABLES FOR STATISTICS
    statstables = [(np.zeros((len(stat_temps), nb)).tolist(), np.zeros((len(stat_temps), nb)).tolist()) for i in range(len(statsfunctions))] # stat_temps[i] -> to be able to have different lists of temperatures for different functions

    # ITERATE
    itermcs = nb * num_in_bin # number of mcs calls + swaps = number of bins * number of values taken into account in a bin
    t_join = 0
    t_spins = 0
    t_tempering = 0
    t_stat = 0
    swaps = [0 for t in range(nt)]
    for it in range(itermcs):
        t1 = time()
        ### MAKE ignoredsteps + consideredsteps
        thread_list = list()
        #create the threads
         # we make EVERYTHING evolve, including threads that we don't measure --> range(nt)
        for t in range(nt):
            if len(saveloops) == nt and saveloops[t] == 1:
                thread_list.append(mcs_withloops(t, iterworm))
            else:
                thread_list.append(mcs_withoutloops(t, iterworm))
        #start the threads
        for t in range(nt):
            thread_list[t].start()
        #wait for all threads to finish
        for t in range(nt):
            thread_list[t].join()
        #return the list of loops
        for t in range(nt):
            if len(saveloops) == nt and saveloops[t] == 1:
                temp_lenloops[t].append(thread_list[t].updatelengths)
                temp_loops[t].append(thread_list[t].updates)
        t2 = time()
        t_join += (t2-t1)/itermcs

        ##parallel tempering
        #take one pair after the other
        for t in range(nt-1, 0, -1):
            #throw a dice
            if (en_states[t] - en_states[t-1]) * (beta_states[t] - beta_states[t-1]) > 0: # if bigger than 0 flip for sure
                #flip accordingly
                #states:
                states[t - 1], states[t] = states[t], states[t - 1]
                #energy:
                en_states[t - 1], en_states[t] = en_states[t], en_states[t - 1]
                swaps[t] += 1
                swaps[t-1] += 1
            elif np.random.uniform() < np.exp((en_states[t] - en_states[t-1]) * (beta_states[t] - beta_states[t-1])): #else flip with a certain prob
                #flip accordingly
                #states:
                states[t - 1], states[t] = states[t], states[t - 1]
                #energy
                en_states[t - 1], en_states[t] = en_states[t], en_states[t - 1]
                swaps[t] += 1
                swaps[t-1] +=1
        t3 = time()
        t_tempering += (t3-t2)/itermcs

        ### STATISTICS UPDATE
        # bin index = integer division of the iteration and the number of iterations in a bin
        #   Before doing any measurement, the spinstate must be updated. But it is not necessary to update the spinstate
        #   if no measurement is performed, since the code runs on dimer configurations.
        #   Hence, we can feed the statistics threads with only the temperatures indices for which we are interested in
        #   the statistics.

        if len(statsfunctions) != 0 or check:
            thread_update_list = list()
            #create threads (NOT range(nt) but stat_temps because we only want to measure SOME states)
            for resid, t in enumerate(stat_temps):
                thread_update_list.append(statistics(t, resid))
            #start updating
            for resid, t in enumerate(stat_temps):
                thread_update_list[resid].start()
            #finish updating
            for resid, t in enumerate(stat_temps):
                thread_update_list[resid].join()
        t4 = time()
        t_spins += (t4-t3)/itermcs
    ### end For


    #verification:
    if len(statsfunctions) != 0 or check:
        for t in stat_temps:
            assert len(onestatecheck(spinstates[t], states[t], d_2s)) == 0, 'Loss of consistency at temperature index {0}'.format(t)
    #optimization info
    print('Time for building loops and saving them = {0}'.format(t_join))
    print('Time for tempering = {0}'.format(t_tempering))
    print('Time for mapping to spins + computing statistics= {0}'.format(t_spins))
    return statstables, swaps



# In[30]:


def onestateevolution(hamiltonian, state, d_nd, d_vd, d_wn, beta,  nitermax, iterworm):

    #true copy
    evolvingstate = np.copy(state)
    energy_differences = []
    updates = []
    updatelengths = []
    savedstates = [state]
    saveloops = True

    for _ in range(iterworm):
        result = dim.manydualworms(hamiltonian, evolvingstate, d_nd, d_vd, d_wn, beta, saveloops, nitermax)
        diff = result[0]
        if diff > 0:
            evolvingstate = np.copy(state)
        else:
            if len(result[1]) != 0:
                energy_differences.append(diff)
                updates.append(np.copy(result[1]))
                updatelengths.append(np.copy(result[2]))
                savedstates.append(np.copy(evolvingstate))

    return energy_differences, updates, updatelengths, savedstates




# In[38]:h


#def neimax_distmax(s_pos, neimax):
#    '''
#        Function returning the distance corresponding to the neimaxth neighbour
#    '''
#    dist = 0
#
#    #create the list of distances
#    distlist = []
#    for s1 in range(len(s_pos)):
#        r1 = s_pos[s1]
#        for s2 in range(len(s_pos)):
#            r2 = s_pos[s2]
#            dist = np.round(np.linalg.norm(r1-r2),3)
#            if dist not in distlist:
#                distlist.append(dist)
#
#    #sort the list of distances
#    distlist.sort()
#
#    if len(distlist) <= neimax :
#        dist = -1
#    else:
#        dist = distlist[neimax]
#
#    return dist

