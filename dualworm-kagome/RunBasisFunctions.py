
# coding: utf-8

# In[1]:

import numpy as np
import dimers as dim
import DualwormFunctions as dw
import StartStates as strst
import Observables as obs

import pickle
import hickle as hkl

from safe import safe

from time import time

import argparse


# In[ ]:

def SafePreparation(args):
    ### PREPARE SAVING
    
    #check that the folder doesn't already exist:
    assert not os.path.exists(args.output + "_folder"), "Folder already exists"
    
    # crate the output directory
    os.mkdir("./" + args.output+"_folder")
    
    # prepare the backup structure
    outputname = "./" + args.output+"_folder/"+"backup"
    
    #if args.safemode == "hickle":
    #    backup = "./" + args.output+"_folder/"+"backup"
    #    hkl.dump("check", backup+".hkl")
    #    mode = False
    #elif args.safemode == "pickle":
    backup = safe()
    backup.params = safe()
    backup.results = safe()
    # check
    pickle.dump(backup, open(outputname + '.pkl','wb'))
    #mode = True
    #else
    #    raise Exception("The safe mode should either be pickle or hickle")
    
    return backup, outputname, mode


# In[ ]:

def SimulationParameters(args, backup, d_ijl,
                         ijl_d, ijl_s, s_ijl, s2_d, L):

    ## Energy
    backup.params.J1 = J1 = args.J1
    backup.params.J2 = J2 = args.J2
    backup.params.J3 = J3 = args.J3
    backup.params.J3st = J3st = J3
    backup.params.J4 = J4 = args.J4
    backup.params.h = h = args.h
    print('J1 ', J1)
    print('J2 ', J2)
    print('J3 ', J3)
    print('J3st ', J3st)
    print('h', h)
    
    ssf = args.ssf
    alternate = args.alternate
    s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)
    
    if ssf:
        print("single spin flip update")
    if alternate:
        print("alternating ssf and dw update")
    
    nnspins, s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)

    
    assert (not (ssf and (J2 != 0 or J3 !=0 or J3st != 0 or J4!=0))),    "The ssf is only available with J1"
    
    updatetype = {'ssf': ssf, 'alternate': alternate,
                  'nnspins': nnspins, 's2p':s2p}
    
    backup.params.updatedtype = updatetype
    
    couplings = {'J1': J1, 'J2':J2, 'J3':J3, 'J3st':J3st, 'J4':J4}
    print("Couplings exacted")
    hamiltonian = dw.Hamiltonian(couplings,d_ijl, ijl_d, L)
    print("hamiltonian computed")
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


    return [couplings, h, hamiltonian, ssf, alternate, s2p, temperatures,
           betas, nt]


# In[ ]:

def StatesAndEnergyInit(args, backup, loadbackup,hamiltonian,
                        nt, d_ijl, d_2s, s_ijl,
                       couplings, h, L):
    if args.loadfromfile:
        print("Initialisation: load from file")
        states = loadbackup.results.states
        spinstates = loadbackup.results.spinstates
        energies = [dw.compute_energy(hamiltonian,states[bid])
                 for bid in range(nt)]
        energies = np.array(energies)
    else:
        ## States
        backup.params.randominit = randominit = args.randominit    
        print('Fully random initialisation = ', randominit)
        backup.params.same = same = args.same
        print('Identical initialisation = ', same)
        backup.params.magninit = magninit = args.magninit
        print('Magnetisation initialisation = ', magninit)


        kwinit = {'random': randominit, 'same': same, 'magninit': magninit, 'h':h}
        print(kwinit)
        print('Same initialisation for all temperatures = ', same)
        (states, energies, spinstates) =        strst.statesinit(nt, d_ijl, d_2s, s_ijl, hamiltonian, **kwinit)

    CheckEnergies(hamiltonian, energies, states, spinstates, h, nt, d_2s)
    
    ref_energies = GSEnergies(couplings, args, L)
    
    return (states, energies, spinstates, ref_energies)


# In[ ]:

def CheckEnergies(hamiltonian, energies, states, spinstates, h, nt, d_2s, checkstates = True):
    
    ## Energies for check
    new_en_states = [dim.hamiltonian(hamiltonian, states[t])-h*spinstates[t].sum() for t in range(nt)]
    new_en_states = np.array(new_en_states)

    for t in range(nt):
        if np.absolute(energies[t]-new_en_states[t]) > 1.0e-5:
            print('RunBasis: Issue at temperature index ', t)
            print("   energies[t] = ", energies[t])
            print("   H0[t] = ", dim.hamiltonian(hamiltonian, states[t]))
            print("   magntot[t] ", spinstates[t].sum())
            print("   new_E[t] = H0[t] - h*magntot[t]", dim.hamiltonian(hamiltonian, states[t]) - h*spinstates[t].sum())

    if checkstates and not dw.statescheck(spinstates, states, d_2s):
        mistakes = [dw.onestatecheck(spinstate, state, d_2s) for spinstate, state in zip(spinstates, states)]
        print('Mistakes: ', mistakes)


# In[ ]:

def CheckGs(args, ref_energies, en_states):
    if args.checkgs:
        t = 0
        if args.checkgsid < len(ref_energies):
            checkgsid = args.checkgsid
            if np.absolute(ref_energies[checkgsid] - en_states[t]) > 1e-10:
                if ref_energies[checkgsid] < en_states[t]:
                    print('RunBasis: Away from gs at t index ', t)
                    print("   en_states[t] = ", en_states[t])
                    print("   ref_energy = ", ref_energies[checkgsid])
                else:
                    print('RunBasis: Energy lower than expected gs at temperature index ', t)
                    print("   en_states[t] = ", en_states[t])
                    print("   ref_energy = ", ref_energies[checkgsid])


# In[ ]:

def GSEnergies(couplings, args,L):
    ref_en = np.zeros(11)
    J1 = couplings['J1']
    J2 = couplings['J2']
    J3 = couplings['J3']
    if args.checkgs:
        # index corresponds to phase description in the ground state of J1-J2-J3 with Kanamori's method
        ref_en[0] = (-2/3 * J1 - 2/3 * J2 + J3)*9*L**2
        ref_en[1] = (-2/3 * J1 - 2/3 * J2 + 3 * J3)*9*L**2
        ref_en[2] = (-2/3 * J1 - 1/3 * J3)*9*L**2
        ref_en[3] = (-2/3 * J1 + 2/3 * J2 - J3)*9*L**2
        ref_en[4] = (-2/3 * J1 + 2 * J2 - J3)*9*L**2
        ref_en[5] = (-2/9 * J1 - 2/3 * J2 - 7/9 * J3)*9*L**2
        ref_en[6] = (-2/15 * J1 - 2/3 * J2 - J3)*9*L**2
        ref_en[7] = (2/3 * J1 - 2/3 * J2 - J3)*9*L**2
        ref_en[8] = (2/3 * J1 - 2/3 * J2 + 1/3 * J3)*9*L**2
        ref_en[9] = (6/7 * J1 - 2/7 * J2 - J3)*9*L**2
        ref_en[10] = (2 * J1 + 2 * J2 + 3 * J3)*9*L**2
        
    return ref_en


# In[ ]:

def ObservablesInit(args, backup, s_ijl, ijl_s, L):
    nnlists = []
    observables = []
    observableslist = []
    energy = args.energy
    if energy:
        observables.append(obs.energy)
        observableslist.append('Energy')
    magnetisation = args.magnetisation
    if magnetisation:
        observables.append(obs.magnetisation)
        observableslist.append('Magnetisation')
        magnfuncid = observableslist.index('Magnetisation')
    else:
        magnfuncid = -1
    charges = args.charges
    if charges:
        observables.append(obs.charges)
        observableslist.append('Charges')
        cfuncid = observableslist.index('Charges')
    else:
        cfuncid = -1
    backup.params.correlations = correlations = args.correlations
    backup.params.all_correlations = all_correlations = args.all_correlations
    backup.params.firstcorrelations = firstcorrelations = args.firstcorrelations
    if correlations:
        observables.append(obs.si)
        observableslist.append('Si')
        if all_correlations:
            observables.append(obs.allcorrelations)
            observableslist.append('All_Correlations')
        else:
            if firstcorrelations:
                print("Check: length of s_ijl", len(s_ijl))
                print("Check: length of NN pairslist:", len(dw.NNpairs(ijl_s, s_ijl, L)))
                print("Check: length of 2ndNN pairs list: ", len(dw.NN2pairs(ijl_s, s_ijl, L)))
                print("Check: length of 3rdparNN pairs list: ", len(dw.NN3parpairs(ijl_s, s_ijl, L)))
                print("Check: length of 3rdstarNN pairs list: ", len(dw.NN3starpairs(ijl_s, s_ijl, L)))
                nnlists = [dw.NNpairs(ijl_s, s_ijl, L), dw.NN2pairs(ijl_s, s_ijl, L),
                           dw.NN3parpairs(ijl_s, s_ijl, L), dw.NN3starpairs(ijl_s, s_ijl, L)]
                observables.append(obs.firstcorrelations)
                observableslist.append('FirstCorrelations')
            else:
                observables.append(obs.centralcorrelations)
                observableslist.append('Central_Correlations')

    print('List of measurements to be performed:', observableslist)
    
    return [nnlists, observables, observableslist, magnfuncid,
            cfuncid]


# In[ ]:

def Temperatures2MeasureInit(args, backup, temperatures, nt):
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
    assert len(stat_temps) <= nt,    'The number of temperature indices to measure cannot be bigger than    the number of temperatures.'
    
    return stat_temps

