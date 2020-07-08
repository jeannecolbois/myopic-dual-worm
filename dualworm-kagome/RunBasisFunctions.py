#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import dimers as dim
import DualwormFunctions as dw
import StartStates as strst
import Observables as obs

import hickle as hkl

import os

from time import time

import argparse


# In[ ]:


def SafePreparation(args):
    '''
        This function makes sure that the location input by the user to save the results is
        available.
    '''
    ### PREPARE SAVING
    
    #check that the folder doesn't already exist:
    assert not os.path.exists(args.output + "_folder"), "Folder already exists"
    
    # crate the output directory
    os.mkdir("./" + args.output+"_folder")
    
    # prepare the backup structure
    backup = "./" + args.output+"_folder/"+"backup"
    hkl.dump("check", backup+".hkl")
    
    return backup


# In[ ]:


def SimulationParameters(args, backup, loadfromfile, d_ijl,
                         ijl_d, ijl_s, s_ijl, s2_d, L):
    
    '''
        This function loads the parameters for the simulation and 
        creates the first few tables - couplings, temperatures,
        magnetic fields
    '''
    ## Energy
    J1 = args.J1
    J2 = args.J2
    J3 = args.J3
    J3st = J3
    J4 = args.J4
    print('J1 ', J1)
    print('J2 ', J2)
    print('J3 ', J3)
    print('J3st ', J3st)
    couplings = {'J1': J1, 'J2':J2, 'J3':J3, 'J3st':J3st, 'J4':J4}
    hkl.dump(couplings,backup+".hkl", path = "/parameters/couplings", mode = 'r+')
    
    print("Couplings extracted")
    hamiltonian = dw.Hamiltonian(couplings,d_ijl, ijl_d, L)
    print("Hamiltonian expression (without field) computed")

    ssf = args.ssf
    alternate = args.alternate
    fullssf = not args.notfullssfupdate

    if ssf:
        print("single spin flip update")
    if alternate:
        print("alternating ssf and dw update")
    if J2 != 0 or J3 != 0 or J4 != 0:
        ssffurther = True
    else:
        ssffurther = False
    nnspins, s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)

    #assert (not ((ssf or alternate) and (J2 != 0 or J3 !=0 or J3st != 0 or J4!=0))),\
    #"The ssf is only available with J1"
    
    updatetype = {'ssf': ssf, 'alternate': alternate,
                  'nnspins': nnspins, 's2p':s2p}
    
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

    mode = args.generatingMode
    return [couplings, hamiltonian, ssf, alternate, ssffurther, s2p, temperatures,
           betas, nt, hfields, nh, mode, fullssf]


# In[ ]:


def StatesAndEnergyInit(args, backup, loadbackup,hamiltonian, ids2walker,
                        nt, nh, hfields, d_ijl, d_2s, s_ijl,
                       couplings, L):
    '''
        This function initialises all the states, either by loading from a file
        or by partially random initialisation based on the user input to the RunBasis function.
        It initialises all the energies as well as the energies to check.
    '''
    
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
    ref_energies, checkgsid = GSEnergies(args, L, couplings)
    if args.checkgs:
        print("Minimal energy index: ", checkgsid, " with energy per site ", ref_energies[checkgsid]/(9*L**2))
    
    return (states, energies, spinstates, ref_energies, checkgsid)


# In[ ]:


def CheckEnergies(hamiltonian, energies, states, spinstates,
                  hfields, nh, ids2walker, nt):
    '''
        This function checks whether the energies computed from the current configurations
        are consistant with the energies computed from the states evolution.
        This is useful for debug of the updates or of the initialisation.
    '''
    ## Energies for check
    new_en_states = [[dim.hamiltonian(hamiltonian,
                                     states[ids2walker[bid,hid]])
                     -hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                     for hid in range(nh)] for bid in range(nt)]
    new_en_states = np.array(new_en_states)

    for t in range(nt):
        for h in range(nh):
            if np.absolute(energies[t,h]-new_en_states[t,h]) > 1.0e-08*np.absolute(new_en_states[t,h]):
                print('RunBasis: Issue at temperature index ', t, ' and h index ', h)
                print("   energies[t,h] = ", energies[t,h])
                print("   H0[t,h] = ", dim.hamiltonian(hamiltonian,
                                                       states[ids2walker[t,h]]))
                print("   magntot[t,h] ", spinstates[ids2walker[t,h]].sum())
                print("   new_E[t,h] = H0[t,h] - h*magntot[t,h]",
                      dim.hamiltonian(hamiltonian, states[ids2walker[t,h]])
                      - hfields[h]*spinstates[ids2walker[t,h]].sum())
                


# In[ ]:


def CheckStates(spinstates, states, d_2s):
    '''
        This function checks whether the given spin states and dimer states are
        indeed compatible.
    '''
    if not dw.statescheck(spinstates, states, d_2s):
        mistakes = [dw.onestatecheck(spinstate, state, d_2s) for spinstate, state in zip(spinstates, states)]
        print('Mistakes: ', mistakes)


# In[ ]:


def CheckGs(args, nt, nh, ref_energies, en_states, checkgsid = -1):
    '''
        If args.checkgs is true, and if the number of hfields is 1,
        this function checks whether the lowest temperature state
        has the expected ground state energy. The expected ground state
        is given by the minimum of the Kanamori index, 
        where this index refers to the numbering
        of the phase obtained from Kanamori's method.
    '''
    ok = True
    if args.checkgs and nh == 1:
        t = 0
        for h in range(nh):
            if checkgsid <0:
                checkgsid = args.checkgsid
            if checkgsid < len(ref_energies):
                if np.absolute(ref_energies[checkgsid] - en_states[t,h]) > 1e-8*np.absolute(ref_energies[checkgsid]):
                    ok = False
                    if ref_energies[checkgsid] < en_states[t,h]:
                        print('RunBasis: Away from gs at t index ', t)
                        print("   en_states[t] = ", en_states[t,h])
                        print("   ref_energy = ", ref_energies[checkgsid])
                    else:
                        print('RunBasis: Energy lower than expected gs at temperature index ', t)
                        print("   en_states[t] = ", en_states[t,h])
                        print("   ref_energy = ", ref_energies[checkgsid])
                        
    return ok


# In[ ]:


def GSEnergies(args, L, couplings):
    '''
        If args.checkgs is true, this function computes the expected ground
        state energy for all of the ground state phases obtained from 
        Kanamori's method for the given system size and couplings 
        and returns them as a list.
        It returns as well the index of the lowest energy.
    '''
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
        
    return ref_en, np.argmin(ref_en)


# In[ ]:


def ObservablesInit(args, backup, s_ijl, ijl_s, L):
    '''
        This function instanciates all the observables that have to be measured.
    '''
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
    correlations = args.correlations
    firstcorrelations = args.firstcorrelations
    both = args.both
    if correlations:
        observables.append(obs.si)
        observableslist.append('Si')

        if firstcorrelations or both:
            nnlists = [dw.NNpairs(ijl_s, s_ijl, L), dw.NN2pairs(ijl_s, s_ijl, L),
                       dw.NN3parpairs(ijl_s, s_ijl, L), dw.NN3starpairs(ijl_s, s_ijl, L)]
            print("Check: length of s_ijl", len(s_ijl))
            print("Check: length of NN pairslist:", len(nnlists[0]))
            print("Check: length of 2ndNN pairs list: ", len(nnlists[1]))
            print("Check: length of 3rd//NN pairs list: ", len(nnlists[2]))
            print("Check: length of 3rd*NN pairs list: ", len(nnlists[3]))

            observables.append(obs.firstcorrelations)
            observableslist.append('FirstCorrelations')
        if (not firstcorrelations) or both:
            observables.append(obs.centralcorrelations)
            observableslist.append('Central_Correlations')

    print('List of measurements to be performed:', observableslist)
    x = (not firstcorrelations) or both;
    obsparams = {'energy':energy, 'magnetisation':magnetisation,
                'charges':charges, 'correlations':correlations,
                 'firstcorrelations':firstcorrelations,
                 'central_correlations':x,
                'observableslist': observableslist}
    hkl.dump(obsparams, backup+".hkl", path = "/parameters/obsparams", mode = 'r+')
    
    return [nnlists, observables, observableslist, magnfuncid,
            cfuncid] 


# In[ ]:


def Temperatures2MeasureInit(args, backup, temperatures, nt):
    '''
        Initialisation of the measurement temperatures
    '''
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
    
    hkl.dump(stat_temps, backup+".hkl", path = "/parameters/stat_temps", mode = 'r+')
    assert len(stat_temps) <= nt,    'The number of temperature indices to measure cannot be bigger than    the number of temperatures.'
    
    return stat_temps


# In[ ]:


def MagneticFields2MeasureInit(args, backup, hfields, nh):
    '''
        Initialisation of the measurement fields
    '''
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

