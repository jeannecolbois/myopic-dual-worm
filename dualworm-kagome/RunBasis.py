
# coding: utf-8

# In[ ]:


import numpy as p
import dimers as dim
import DualwormFunctions as dw
import StartStates as strst
import Observables as obs

import pickle
from safe import safe

import argparse


# In[ ]:


### PARSING
parser = argparse.ArgumentParser()

parser.add_argument('--L', type = int, default = 4, help = 'Lattice side size')

# COUPLINGS
parser.add_argument('--J1', type = float, default = 1.0,
                    help = 'NN coupling') # nearest-neighbour coupling
parser.add_argument('--J2', type = float, default = 0.0,
                    help = '2nd NN coupling') # 2nd NN coupling
parser.add_argument('--J3', type = float, default = 0.0,
                    help = '3rd NN coupling') # 3rd NN coupling
parser.add_argument('--J4', type = float, default = 0.0,
                    help = '4th NN coupling')

#NUMBER OF STEPS AND ITERATIONS
parser.add_argument('--nst', type = int, default = 100,
                    help = 'number of thermalisation steps') # number of thermalisation steps
parser.add_argument('--nsm', type = int, default = 100,
                    help = 'number of measurements steps') # number of measurement steps
parser.add_argument('--nips', type = int, default = 10,
                    help = 'number of worm constructions per MC step')
parser.add_argument('--nb', type = int, default = 20,
                    help = 'number of bins')

#PARALLELISATION
parser.add_argument('--nthreads', type = int, default = 4,
                    help = 'number of threads to use')

#WORM PARAMETERS
parser.add_argument('--nmaxiter', type = int, default = 10,
                    help = '''maximal number of segments in a loop update over the
                    size of the lattice (1 = 1times the number of dualbonds in the
                    lattice)''')
parser.add_argument('--randominit', default = False, action ='store_true',
                    help = 'intialise the states randomly')
parser.add_argument('--same', default = False, action = 'store_true',
                    help = '''initialise all temperatures with the same
                    state (debug purposes)''')

#TEMPERATURE PARAMETERS
parser.add_argument('--t_list', nargs = '+', type = float, default = [0.5, 15.0],
                    help = 'list of limiting temperature values')
parser.add_argument('--nt_list', nargs = '+', type = int, default = [28],
                    help = 'list of number of temperatures in between the given limiting temperatures')
parser.add_argument('--log_tlist', default = False, action='store_true',
                    help = 'state whether you want the temperature be spaced log-like or linear-like (activate if you want log)')
parser.add_argument('--stat_temps_lims', nargs = '+', type = float,
                    help = '''limiting temperatures for the various ranges of
                    measurements''') 
                    #default will be set to none, and then we can decide what to do later on.

#CORRELATIONS PARAMETER
parser.add_argument('--energy', default = False, action = 'store_true',
                    help = 'activate if you want to save the energy')
parser.add_argument('--magnetisation', default = False, action = 'store_true',
                    help = 'activate if you want to save the magnetisation')
parser.add_argument('--correlations', default = False, action = 'store_true',
                    help = 'activate if you want to save either central or all correlations')
parser.add_argument('--all_correlations', default = False, action = 'store_true',
                    help = '''activate if you want to save the correlations for all non-equivalent
                    pairs of sites. Otherwise, will save central correlations.''')
#SAVE
parser.add_argument('--output', type = str, help = 'saving filename (.pkl will be added)')
args = parser.parse_args()


# In[ ]:


### PREPARE SAVING
backup = safe()
backup.params = safe()
backup.results = safe()


# In[ ]:


### SIMULATIONS INITIATLISATION
backup.params.L = L = args.L
print('Lattice side size: ', L)
[d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, 
 d_nd, d_vd, d_wn, sidlist, didlist] = dw.latticeinit(L)

## Energy
backup.params.J1 = J1 = args.J1
backup.params.J2 = J2 = args.J2
backup.params.J3 = J3 = args.J3
backup.params.J3st = J3st = J3
backup.params.J4 = J4 = args.J4
print('J1 ', J1)
print('J2 ', J2)
print('J3 ', J3)
print('J3st ', J3st)
print('J4', J4)

couplings = {'J1': J1, 'J2':J2, 'J3':J3, 'J3st':J3st, 'J4':J4}
hamiltonian = dw.Hamiltonian(couplings,d_ijl, ijl_d, L)

## Temperatures to simulate
t_list = [t for t in args.t_list]
nt_list = args.nt_list
backup.params.loglist = loglist = args.log_tlist
if loglist:
    temperatures = dw.create_log_temperatures(nt_list, t_list)
else:
    temperatures = dw.create_temperatures(nt_list, t_list)
betas = 1/temperatures
backup.params.temperatures = temperatures
backup.params.nt = nt = len(temperatures) # total number of different temperatures
print('Number of temperatures: ', nt)

## States
backup.params.randominit = randominit = args.randominit    

print('Random initialisation = ', randominit)

backup.params.same = same = args.same

print('Same initialisation for all temperatures = ', same)
    #kf.statesinit(number of temperatures, dual bond table, spin surrounding dual bonds, spin site table, hamiltonian list, random starting state, same type of starting state for all temperatures)
(states, energies) = strst.statesinit(nt, d_ijl, d_2s, s_ijl, hamiltonian, randominit, same)


spinstates = dw.states_dimers2spins(sidlist, didlist, L, states)
new_en_states = [dim.hamiltonian(hamiltonian, states[t]) for t in range(nt)]
for t in range(nt):
    if np.absolute(energies[t]-new_en_states[t]) > 1.0e-5:
        print('RunBasis: Issue at temperature index', t)
if not kf.statescheck(spinstates, states, d_2s):
    mistakes = [dw.onestatecheck(spinstate, state, d_2s) for spinstate, state in zip(spinstates, states)]
    print('Mistakes: ', mistakes)


# In[ ]:


### INITIALISATION FOR THE MEASUREMENTS

# Observables to measure
observables = []
observableslist = []
backup.params.energy = energy = args.energy
if energy:
    observables.append(obs.energy)
    observableslist.append('Energy')
backup.params.magnetisation = magnetisation = args.magnetisation
if magnetisation:
    observables.append(obs.magnetisation)
    observableslist.append('Magnetisation')
backup.params.correlations = correlations = args.correlations
backup.params.all_correlations = all_correlations = args.all_correlations
if correlations:
    observables.append(obs.si)
    observableslist.append('Si')
    if all_correlations:
        observables.append(obs.allcorrelations)
        observableslist.append('All_Correlations')
    else:
        observables.append(obs.centralcorrelations)
        observableslist.append('Central_Correlations')
        
print('List of measurements to be performed:', observableslist)

# Temperatures to measure
if args.stat_temps_lims is None:
    #by default, we measure the whole temperature range
    stat_temps = range(nt)
else: # if args.stat_temps is not none
    vals = []
    stat_temps = []
    # we need to turn the stat_temps_lims into actual lists of indices
    for val in args.stat_temps_lims:
        vals.append(np.abs(temperatures-val).argmin())
        print(val, vals)
    l = len(vals)
    assert(l%2 == 0)
    for i in range(0, l, 2):
        stat_temps += list(range(vals[i], vals[i+1]+1))
        
print('List of temperature indices to measure:', stat_temps)
backup.params.stat_temps = stat_temps
assert len(stat_temps) <= nt, 'The number of temperature indices to measure cannot be bigger than the number of temperatures.'


## THERMALISATION
#preparation
nb = 1 # only one bin, no statistics
num_in_bin = args.nst# mcs iterations per bins
iterworm = nips = args.nips #number of worm iterations before considering swaps
nmaxiter = args.nmaxiter
statsfunctions = [] #don't compute any statistics
check = 0 #don't turn to spins to check
print('Number of thermalisation steps = ', num_in_bin*nb)
backup.params.thermsteps = num_in_bin*nb
backup.params.nthreads = nthreads = args.nthreads
#launch thermalisation
#states = list(states)
t1 = time()
(meanstatth, swapsth) = dw.mcs_swaps(nb, num_in_bin, iterworm, check, statsfunctions, nt, stat_temps, hamiltonian, d_nd, d_vd, d_wn, d_2s, s2_d, sidlist, didlist, L, states, spinstates, energies, betas, s_ijl, nmaxiter, nthreads)
t2 = time()
#states = np.array(states)
backup.results.swapsth = swapsth
print('Time for all thermalisation steps = ', t2-t1)

spinstates = dw.states_dimers2spins(sidlist, didlist, L, states)
new_en_states = [dim.hamiltonian(hamiltonian, states[t]) for t in range(nt)]
for t in range(nt):
    if np.absolute(energies[t]-new_en_states[t]) > 1.0e-5:
        print('RunBasis: Issue at temperature index', t)


## MEASUREMENT PREPARATION 

# Preparation to call the method
backup.params.nb = nb = args.nb # number of bins
backup.params.num_in_bin = num_in_bin = args.nsm//nb
print('Number of measurement steps = ', num_in_bin*nb) # number of iterations = nb * num_in_bin 
iterworm = nips #number of worm iterations before considering swaps and measuring the state again
statsfunctions = observables #TODO set functions
backup.results.namefunctions = observableslist #TODO set functions corresponding to the above
print(backup.results.namefunctions)
check = 1 #turn to spins and check match works

#states = list(states)
# Run measurements
t1 = time()
(backup.results.meanstat, backup.results.swaps) = (meanstat, swaps) = dw.mcs_swaps(nb, num_in_bin, iterworm, check, statsfunctions, nt, stat_temps, hamiltonian, d_nd, d_vd, d_wn, d_2s, s2_d, sidlist, didlist, L, states, spinstates, energies, betas, s_ijl, nmaxiter, nthreads)
t2 = time()
print('Time for all measurements steps = ', t2-t1)

#states = np.array(states)
spinstates = dw.states_dimers2spins(sidlist, didlist, L, states)
new_en_states = [dim.hamiltonian(hamiltonian, states[t]) for t in range(nt)]
for t in range(nt):
    if np.absolute(energies[t]-new_en_states[t]) > 1.0e-5:
        print('RunBasis: Issue at temperature index', t)


## STATISTICS ##

t_meanfunc = list() #for each function, for each temperature, mean of the state function
t_varmeanfunc = list() #for each function, for each temperature, variance of the state function
numsites = len(s_ijl)

for idtuple, stattuple in enumerate(meanstat):
    # means:
    t_meanfunc.append((np.array(stattuple[0]).sum(1)/nb, np.array(stattuple[1]).sum(1)/nb))

    #variances:
    tuplevar1 = [0 for t in stat_temps]
    tuplevar2 = [0 for t in stat_temps]
    for resid, t in enumerate(stat_temps):
        for b in range(nb):
            tuplevar1[resid] += ((stattuple[0][resid][b] - t_meanfunc[idtuple][0][resid]) ** 2)/(nb * (nb - 1))
            tuplevar2[resid] += ((stattuple[1][resid][b] - t_meanfunc[idtuple][1][resid]) ** 2)/(nb * (nb - 1))
    t_varmeanfunc.append((tuplevar1, tuplevar2))

# Additional results for the correlations are handled directly in AnalysisBasis_3dot1dot5

#Save the final results
backup.results.t_meanfunc = t_meanfunc
backup.results.t_varmeanfunc = t_varmeanfunc
backup.results.states = states
backup.results.spinstates = spinstates
#Save the backup object in a file
pickle.dump(backup, open(args.output + '.pkl','wb'))
print('Job done')

