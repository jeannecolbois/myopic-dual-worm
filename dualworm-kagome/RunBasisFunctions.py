
# coding: utf-8

# In[ ]:


import numpy as np
import dimers as dim
import DualwormFunctions as dw
import StartStates as strst
import Observables as obs

import pickle
import hickle as hkl

from safe import safe

import os

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
    backup = "./" + args.output+"_folder/"+"backup"
    hkl.dump("check", backup+".hkl")
    
    return backup


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

    ## Energy
    J1 = args.J1
    J2 = args.J2
    J3 = args.J3
    J3st = J3
    J4 = args.J4
    h = args.h
    print('J1 ', J1)
    print('J2 ', J2)
    print('J3 ', J3)
    print('J3st ', J3st)
    print('h', h)
    
    
    couplings = {'J1': J1, 'J2':J2, 'J3':J3, 'J3st':J3st, 'J4':J4}
    hkl.dump(couplings,backup+".hkl", path = "/parameters/couplings", mode = 'r+')
    
    print("Couplings extracted")
    hamiltonian = dw.Hamiltonian(couplings,d_ijl, ijl_d, L)
    print("Hamiltonian expression (without field) computed")

    ssf = args.ssf
    alternate = args.alternate

    s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)
    if ssf:
        print("single spin flip update")
    if alternate:
        print("alternating ssf and dw update")
    nnspins, s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)

    
    assert (not ((ssf or alternate) and (J2 != 0 or J3 !=0 or J3st != 0 or J4!=0))),    "The ssf is only available with J1"
    
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
    hkl.dump(updatetype,backup+".hkl", path = "/parameters/updatetype", mode = 'r+')
    

    ## Temperatures and magnetic fields to simulate
    t_list = [t for t in args.t_list]
    nt_list = args.nt_list
    loglist = args.log_tlist
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
    # total number of different temperatures
    nt = len(temperatures)
    if args.h_list:
        h_list = [h for h in args.h_list]
        nh_list = args.nh_list
        hfields = dw.create_hfields(nh_list, h_list)
        nh = len(hfields)
    else:
        hfields = np.array([h])
        nh = 1
    
    physical = {'temperatures': temperatures, 'nt':nt, 'nh':nh,
                'hfields': hfields}
    hkl.dump(physical,backup+".hkl", path = "/parameters/physical", mode = 'r+')
    
    print('Number of temperatures: ', nt)
    print('Temperatures:', temperatures)
    print('Number of magnetic fields: ', nh)
    print('Magnetic fields: ', hfields)

    return [couplings, hamiltonian, ssf, alternate, s2p, temperatures,
           betas, nt, hfields, nh]


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

def StatesAndEnergyInit(args, backup, loadbackup,hamiltonian, ids2walker,
                        nt, nh, hfields, d_ijl, d_2s, s_ijl,
                       couplings, L):
    if args.loadfromfile:
        print("Initialisation = loading from file!")
        states = hkl.load(loadbackup+"/backup_states.hkl")
        spinstates = hkl.load(loadbackup+"/backup_spinstates.hkl")
        energies = [[dw.compute_energy(hamiltonian,states[ids2walker[bid, hid]])
                  - hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                  for hid in range(nh)]
                 for bid in range(nt)]
    
        energies = np.array(energies)
    else:
        randominit = args.randominit    
        print('Fully random initialisation = ', randominit)
        same = args.same
        print('Identical initialisation = ', same)
        magninit = args.magninit
        print('Magnetisation initialisation = ', magninit)
        maxflip = args.maxflip
        magnstripes = args.magnstripes

        kwinit = {'random': randominit, 'same': same, 
                  'magninit': magninit, 'maxflip':maxflip,
                 'magnstripes': magnstripes}

        hkl.dump(kwinit, backup+".hkl", path = "/parameters/kwinit", mode = 'r+')
        
        (states, energies,spinstates) =        strst.statesinit(nt, nh, hfields, ids2walker, d_ijl,
                         d_2s, s_ijl, hamiltonian, **kwinit)
        
    CheckEnergies(hamiltonian, energies, states, spinstates,
                 hfields, nh, ids2walker, nt)
    # Energy states of reference
    ref_en_states = np.zeros(len(hfields))
    J1 = couplings['J1']
    J2 = couplings['J2']
    J3 = couplings['J3']
    if args.checkgs:
        for hid, h in enumerate(hfields):
            if h == 0:
                if J2 == 0:
                    ref_en_states[hid] = 9*L**2*(-2/3 * J1 - J3)
                elif J3/J2 < 0.5:
                    ref_en_states[hid] = 9*L**2*(-2/3 * J1 - 2/3 * J2)
                elif J3/J2 >= 0.5 and J3/J2 < 1:
                    ref_en_states[hid] = 9*L**2*(-2/3 * J1 - 1/3 * J3)
                elif J3/J2 >= 1:
                    ref_en_states[hid] = 9*L**2*(-2/3 * J1 + 2/3 * J2 - J3)
            elif abs(h/J1) < 4:
                ref_en_states[hid] = 9*L**2*(-2/3 * J1 - 1/3 * abs(h))
            elif abs(h/J1) > 4:
                ref_en_states[hid] = 9*L**2*(2*J1 - abs(h))
        print(ref_en_states)
    else:
        ref_en_states = np.array([])
        
    return (states, energies, spinstates, ref_en_states)


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

def CheckEnergies(hamiltonian, energies, states, spinstates,
                  hfields, nh, ids2walker, nt):
    
    ## Energies for check
    new_en_states = [[dim.hamiltonian(hamiltonian,
                                     states[ids2walker[bid,hid]])
                     -hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                     for hid in range(nh)] for bid in range(nt)]
    new_en_states = np.array(new_en_states)

    for t in range(nt):
        for h in range(nh):
            if np.absolute(energies[t,h]-new_en_states[t,h]) > 1.0e-09:
                print('RunBasis: Issue at temperature index ', t, ' and h index ', h)
                print("   energies[t,h] = ", energies[t,h])
                print("   H0[t,h] = ", dim.hamiltonian(hamiltonian,
                                                       states[ids2walker[t,h]]))
                print("   magntot[t,h] ", spinstates[ids2walker[t,h]].sum())
                print("   new_E[t,h] = H0[t,h] - h*magntot[t,h]",
                      dim.hamiltonian(hamiltonian, states[ids2walker[t,h]])
                      - hfields[h]*spinstates[ids2walker[t,h]].sum())


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
    correlations = args.correlations
    all_correlations = args.all_correlations
    firstcorrelations = args.firstcorrelations
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
    
    obsparams = {'energy':energy, 'magnetisation':magnetisation,
                'charges':charges, 'correlations':correlations,
                'all_correlations':all_correlations,
                 'firstcorrelations':firstcorrelations,
                'observableslist': observableslist}
    hkl.dump(obsparams, backup+".hkl", path = "/parameters/obsparams", mode = 'r+')
    
    return [nnlists, observables, observableslist, magnfuncid,
            cfuncid] 


# In[ ]:


def Temperatures2MeasureInit(args, backup, temperatures, nt):
    if args.stat_temps_lims is None:
        #by default, we measure all the temperatures
        stat_temps = list(range(nt))
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
    hkl.dump(stat_temps, backup+".hkl", path = "/parameters/stat_temps", mode = 'r+')
    assert len(stat_temps) <= nt,    'The number of temperature indices to measure cannot be bigger than    the number of temperatures.'
    
    return stat_temps


# In[ ]:


def MagneticFields2MeasureInit(args, backup, hfields, nh):
    # Magnetic fields to measure
    if args.stat_hfields_lims is None:
        #by default, we measure all the temperatures
        stat_hfields = list(range(nh))
    else: # if args.stat_temps is not none
        vals = []
        stat_hfields = []
        # we need to turn the stat_temps_lims into actual lists of indices
        for val in args.stat_hfields_lims:
            vals.append(np.abs(hfields-val).argmin())
            print(val, vals)
        l = len(vals)
        assert(l%2 == 0)
        for i in range(0, l, 2):
            stat_hfields += list(range(vals[i], vals[i+1]+1))
    
    print('List of field indices to measure:', stat_hfields)
    
    hkl.dump(stat_hfields, backup+".hkl", path = "/parameters/stat_hfields", mode = 'r+')
    assert len(stat_hfields) <= nh,    'The number of field indices to measure cannot be bigger than    the number of fields.'
    
    return stat_hfields


# In[ ]:


def CheckGs(args, ref_en_states, new_en_states, nh):
    if args.checkgs:
        t = 0
        for h in range(nh):
            if np.absolute(ref_en_states[h]-new_en_states[t,h]) > 1.0e-12:
                print('RunBasis: Away from gs at t index ', t, ' and h index ', h)
                print("   new_en_states[t,h] = ", new_en_states[t,h])
                print("   ref_en_states[h] = ", ref_en_states[h])

