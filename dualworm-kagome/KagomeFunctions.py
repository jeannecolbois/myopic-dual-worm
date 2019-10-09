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

