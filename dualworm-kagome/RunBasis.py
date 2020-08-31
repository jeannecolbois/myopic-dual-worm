#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import dimers as dim
import DualwormFunctions as dw
import StartStates as strst
import Observables as obs
import RunBasisFunctions as rbf

import hickle as hkl
from safe import safe
import os
import warnings

from time import time

import argparse


# In[ ]:


def main(args):
    verbose = args.verbose
    print("-------------------Initialisation--------------------")
    ### PREPARE SAVING
    backup = rbf.SafePreparation(args)

    print(backup+".hkl")

    ### SIMULATIONS PARAMETERS
    loadfromfile = args.loadfromfile

    if loadfromfile:
        loadbackup = args.filename
        L = hkl.load(loadbackup+"/backup.hkl", path = "/parameters/L")
        assert L == args.L, "Loaded and required lattice sizes not compatible."
    else:
        loadbackup = ""

    L = args.L
    hkl.dump(L, backup+".hkl", path="/parameters/L", mode = 'r+')

    print('Lattice side size: ', L)
    [d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, d_nd, d_vd, d_wn,
     sidlist, didlist, c_ijl, ijl_c, c2s, csign] =\
    dw.latticeinit(L)

    [couplings, hamiltonian, ssf, alternate, ssffurther, s2p, temperatures,
     betas, nt, hfields, nh, genMode, fullssf] =\
    rbf.SimulationParameters(args,backup, loadfromfile, d_ijl,
                             ijl_d, ijl_s, s_ijl, s2_d, L)

    if loadfromfile:
        try:
            walker2params = hkl.load(loadbackup+"/backup_walker2params.hkl")
        except OSError:
            pass
        
        try:
            walker2ids = hkl.load(loadbackup+"/backup_walker2ids.hkl")
        except OSError:
            pass
        
        try:
            ids2walker = hkl.load(loadbackup+"/backup_ids2walker.hkl")
        except OSError:
            warnings.warn("error ids2walker load")
            [walker2params, walker2ids, ids2walker] =            dw.walkerstable(betas, nt, hfields, nh)
            
       
    else:
        [walker2params, walker2ids, ids2walker] =        dw.walkerstable(betas, nt, hfields, nh)
    print("ids2walker shape", ids2walker.shape)
    # Saving the status:
    
    ## SIMULATION INITIALISATION
    (states, energies, spinstates, ref_energies, checkgsid)=    rbf.StatesAndEnergyInit(args, backup, loadbackup,hamiltonian,
                            ids2walker, nt, nh, hfields, d_ijl,
                            d_2s, s_ijl, couplings, L)
    # Checks
    ok = rbf.CheckGs(args, nt, nh, ref_energies, energies, checkgsid)
    
    rbf.CheckEnergies(hamiltonian, energies, states, spinstates,
                  hfields, nh, ids2walker, nt)
    print("Energies Checked")
    
    rbf.CheckStates(spinstates, states, d_2s)
    print("States Checked")

    ### INITIALISATION FOR THE MEASUREMENTS
    # Observables to measure
    [nnlists, observables, observableslist, magnfuncid,cfuncid,srefs]  =    rbf.ObservablesInit(args, backup, s_ijl, ijl_s, L)

    # Temperatures to measure
    stat_temps = rbf.Temperatures2MeasureInit(args, backup,
                                             temperatures, nt)

    # Magnetic fields to measure
    stat_hfields = rbf.MagneticFields2MeasureInit(args, backup,
                                                 hfields, nh)

    ## THERMALISATION
    #preparation
    ncores = args.ncores

    print("-----------Thermalisation------------------")
    nb = 1 # only one bin, no statistics
    num_in_bin = args.nst# mcs iterations per bins
    iterworm = nips = args.nips # maximal number of MCS iterations
    # before considering swaps (one iteration = one system size update)
    nrps = args.nrps # number of replica loop iterations in a MC step
    nmaxiter = args.nmaxiter
    statsfunctions = [] #don't compute any statistics
    check = False #don't turn to spins to check
    print('Number of thermalisation steps = ', num_in_bin*nb)
    thermsteps = num_in_bin*nb
    #launch thermalisation
    thermalisation = {'ncores':ncores, 'nmaxiter':nmaxiter,
                     'thermsteps': thermsteps}
    hkl.dump(thermalisation,backup+".hkl",
             path = "/parameters/thermalisation", mode = 'r+')

    kw = {'nb':nb,'num_in_bin':num_in_bin, 'iterworm':iterworm,
          'nrps': nrps,
          'nitermax':nmaxiter,'check':check,
          'statsfunctions':statsfunctions,
          'nt':nt, 'hamiltonian':hamiltonian,'ncores':ncores,
          'd_nd':d_nd,'d_vd':d_vd,'d_wn':d_wn, 'd_2s':d_2s, 's2_d':s2_d,
          'sidlist':sidlist,'didlist':didlist,'s_ijl':s_ijl,'ijl_s':ijl_s,
          'L':L, 'nh':nh, 'hfields':hfields,
          'walker2params':walker2params,'walker2ids':walker2ids,
          'ids2walker':ids2walker,
          's2p':s2p, 'ssf':ssf, 'ssffurther': ssffurther,
          'alternate':alternate, 'fullstateupdate': True,
          'verbose':verbose
         }


    t1 = time()
    (statstableth, swapst_th, swapsh_th, failedupdatesth, 
     failedssfupdatesth, updateliststh) = \
    dw.mcs_swaps(states, spinstates, energies, betas, [],[], **kw)
    t2 = time()

    print('Time for all thermalisation steps = ', t2-t1)


    thermres = {'swapst_th':swapst_th, 'swapsh_th':swapsh_th,
               'failedupdatesth':failedupdatesth,
                'failedssfupdatesth':failedssfupdatesth, 'totaltime':t2-t1}
    hkl.dump(thermres,backup+".hkl",
             path = "/results/thermres", mode = 'r+')

    # Checks
    ok = rbf.CheckGs(args, nt, nh, ref_energies, energies, checkgsid)
    
    rbf.CheckEnergies(hamiltonian, energies, states, spinstates,
                  hfields, nh, ids2walker, nt)
    print("Energies Checked")
    
    rbf.CheckStates(spinstates, states, d_2s)
    print("States Checked")
    measupdate = args.measupdate
    
    if genMode and not ok and not measupdate:
        # save status and throw an error
        hkl.dump(states, backup+"_states.hkl")
        hkl.dump(spinstates, backup+"_spinstates.hkl")
        hkl.dump(walker2params, backup+"_walker2params.hkl")
        hkl.dump(walker2ids, backup+"_walker2ids.hkl")
        hkl.dump(ids2walker, backup+"_ids2walker.hkl")
        print("/!\ Problem with generating mode ...States and spinstates saved")
        raise Exception('''The generating Mode is activated
        but the states are not in the ground state after thermalisation.
        Try re-running the thermalisation procedure starting from the current states,
        which have been saved.''')
    elif not ok:
        warnings.warn("Lowest temperature state not in the ground state after thermalisation")
    ## MEASUREMENT PREPARATION 
    
    print("-----------Measurements-----------------")
    # Preparation to call the method
    nb = args.nb # number of bins
    num_in_bin = args.nsm//nb
    print('Number of measurement steps = ', num_in_bin*nb) # number of iterations = nb * num_in_bin 
    iterworm = nips #number of worm iterations before considering swaps and measuring the state again
    statsfunctions = observables #TODO set functions
    namefunctions = observableslist #TODO set functions corresponding to the above
    print(namefunctions)
    check = True #turn to spins and check match works
    measperiod = args.measperiod
    print('Measurement period:', measperiod)
    
    
    if measupdate:
        nnspins, s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)
        htip = args.htip
        Ttip = args.Ttip
        pswitch = args.pswitch
        uponly = args.uponly
        measupdatev = args.measupdatev
        saveupdates = args.saveupdates
        path = dw.path_for_measupdate(s_ijl, ijl_s, s2_d, L, version = measupdatev)
    else:
        if not (ssf or alternate):
            s2p = []
            
        nnspins = []
        htip = 0
        Ttip = 0
        measupdatev = 0
        saveupdates = False
        pswitch = 1
        uponly = False
        path = []

    kwmeas = {'nb':nb, 'num_in_bin':num_in_bin,'nips':nips,
              'nrps':nrps,
              'measperiod':measperiod, 
              'measupdate':measupdate, 'measupdatev' : measupdatev,
              'htip': htip, 'Ttip':Ttip, 'pswitch': pswitch, 'uponly': uponly,
              'nnspins':nnspins, 's2p': s2p}
    hkl.dump(kwmeas, backup+".hkl", path = "/parameters/measurements", mode = 'r+')
    if verbose:
        print("uponly: ", uponly)
    kw = {'nb':nb,'num_in_bin':num_in_bin, 'iterworm':iterworm,
          'nrps': nrps,
          'nitermax':nmaxiter,'check':check,
          'statsfunctions':statsfunctions,
          'nt':nt, 'hamiltonian':hamiltonian,
          'nnlists':nnlists,
          'd_nd':d_nd,'d_vd':d_vd,'d_wn':d_wn, 'd_2s':d_2s, 's2_d':s2_d,
          'sidlist':sidlist,'didlist':didlist,'s_ijl':s_ijl,'ijl_s':ijl_s,'L':L,
          'ncores':ncores, 
          'measupdate': measupdate, 'measupdatev' : measupdatev, 'path': path,
          'htip':htip, 'Ttip':Ttip,'pswitch': pswitch, 'uponly':uponly,
          'nnspins': nnspins, 's2p':s2p,
          'magnfuncid':magnfuncid,
          'c2s':c2s, 'csign':csign, 'measperiod':measperiod,
          'nh':nh, 'hfields':hfields, 'walker2params':walker2params,
          'walker2ids':walker2ids,'ids2walker':ids2walker,
          'ssf':ssf, 'ssffurther': ssffurther, 
          'alternate':alternate, 'randspinupdate': False,
          'namefunctions': namefunctions, 'srefs':srefs,
          'backup': backup,
          'genMode': genMode, 'fullstateupdate': fullssf,
          'saveupdates': saveupdates,
          'verbose':verbose}
        # Run measurements

    t1 = time()
    (statstable, swapst, swapsh, failedupdates, failedssfupdates, updatelists) =    dw.mcs_swaps(states, spinstates, energies, betas, stat_temps, stat_hfields,**kw)
    #print("Energies = ", energies)
    t2 = time()

    print('Time for all measurements steps = ', t2-t1)
    print("Energies size: ", energies.shape)

    totupdates = nips*num_in_bin*nb*measperiod*len(s_ijl)
    measurementsres = {'swapst': swapst, 'swapsh': swapsh,
                       'failedupdates':failedupdates,'totupdates':totupdates, 
                       'failedssfupdates':failedssfupdates, 'updatelists':updatelists}

    hkl.dump(measurementsres, backup+".hkl", path = "/results/measurements", mode = 'r+')

    # Checks
    rbf.CheckGs(args, nt, nh, ref_energies, energies, checkgsid)
    
    rbf.CheckEnergies(hamiltonian, energies, states, spinstates,
                  hfields, nh, ids2walker, nt)
    print("Energies Checked")
    
    rbf.CheckStates(spinstates, states, d_2s)
    print("States Checked")
    
    #Save the final results and the order in which to read them...
    hkl.dump(walker2params, backup+"_walker2params.hkl")
    hkl.dump(walker2ids, backup+"_walker2ids.hkl")
    hkl.dump(ids2walker, backup+"_ids2walker.hkl")
    
    print("ids2walker shape", ids2walker.shape)
    hkl.dump(states, backup+"_states.hkl")
    hkl.dump(spinstates, backup+"_spinstates.hkl")
    
    print("Job done")
    return statstable, swapst, swapsh, failedupdatesth, failedupdates, failedssfupdates


# In[ ]:


if __name__ == "__main__":

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
    parser.add_argument('--J3st', type = float, default=argparse.SUPPRESS,
                       help = '3rd star NN coupling. If not given, set to J3 value. If given, can be different from J3.')
    parser.add_argument('--J4', type = float, default = 0.0,
                        help = '4th NN coupling')
    
    #NUMBER OF STEPS AND ITERATIONS
    parser.add_argument('--nst', type = int, default = 100,
                        help = 'number of thermalisation steps') # number of thermalisation steps
    parser.add_argument('--nsm', type = int, default = 100,
                        help = 'number of measurements steps') # number of measurement steps
    parser.add_argument('--nips', type = int, default = 10,
                        help = 'number of worm constructions per MC step')
    parser.add_argument('--nrps', type = int, default = 1,
                        help = 'number of replica loops per MC step')
    parser.add_argument('--measperiod', type = int, default = 1,
                        help = 'number of nips worm building + swaps between measurements')
    parser.add_argument('--nb', type = int, default = 20,
                        help = 'number of bins')

    #PARALLELISATION
    parser.add_argument('--ncores', type = int, default = 4,
                        help = 'number of threads to use')

    # INITIALISATION PARAMETERS
    parser.add_argument('--randominit', default = False, action ='store_true',
                        help = 'intialise the states randomly')
    parser.add_argument('--same', default = False, action = 'store_true',
                        help = '''initialise all temperatures with the same
                        state (debug purposes)''')
    parser.add_argument('--testinit', default = False, action = 'store_true',
                        help = '''test some given initialisation''')
    parser.add_argument('--magninit', default = False, action = 'store_true',
                        help = '''initialise all the temperature with
                        one of the m=1/3 GS''')
    parser.add_argument('--magnstripes', default = False, action = 'store_true',
                       help = '''initialise all the temperature with
                       m=1/3 stripes''')
    parser.add_argument('--maxflip', default = False, action = 'store_true',
                       help = '''initialise all the temperature with
                       maximally flippable plateau''')
    parser.add_argument('--loadfromfile', default = False, action = 'store_true',
                       help = '''initialise all the states with
                       results from a previously performed simulations''')
    parser.add_argument('--filename', type = str, default = "", help = '''initialise all the states with
                       results from a previously performed simulations''')
    
    #WORM PARAMETERS
    parser.add_argument('--nmaxiter', type = int, default = 10,
                        help = '''maximal number of segments in a loop update over the
                        size of the lattice (1 = 1times the number of dualbonds in the
                        lattice)''')
    parser.add_argument('--measupdate', default = False, action = 'store_true',
                       help = '''activate to mimic the action of the measuring tip''')
    parser.add_argument('--measupdatev', type = int, default = 0,
                       help = '''select the version of measupdate''')
    parser.add_argument('--saveupdates', default = False, action = 'store_true',
                       help = '''activate to save the effect of the measuring tip (only genMode)''')
    parser.add_argument('--htip', type = float, default = 0.0, 
                       help = '''magnetic field associated with the tip''')
    parser.add_argument('--Ttip', type = float, default = 0.0, 
                       help = '''temperature associated with the tip measurements''')
    parser.add_argument('--pswitch', type = float, default = 1, 
                       help = '''tip switching probability''')
    parser.add_argument('--uponly', default = False, action = 'store_true',
                       help = '''Only switching down spins to up spins in measupdate''')
    parser.add_argument('--ssf', default = False, action = 'store_true',
                        help = 'activate for single spin flip update')
    parser.add_argument('--notfullssfupdate', default = False, action = 'store_true',
                        help = 'whether to fully update the state or not at each ssf step *during the measurement phase*')
    parser.add_argument('--alternate', default = False, action = 'store_true',
                        help = 'activate for single spin flip update and dw update')
    parser.add_argument('--checkgs', default = False, action = 'store_true',
                        help = 'check wether the simulation reaches the expected ground state')
    parser.add_argument('--checkgsid', type = int, default = 0,
                        help = 'index of the ground state phase to check')
    
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
    
    # MAGNETIC FIELD PARAMETERS
    parser.add_argument('--h_list', nargs = '+', type = float, default = [0,0],
                        help = 'list of limiting magnetic field values')
    parser.add_argument('--nh_list', nargs = '+', type = int, default = [1],
                        help = 'list of number of magnetic fields in between the given limiting temperatures')
    parser.add_argument('--stat_hfields_lims', nargs = '+', type = float,
                    help = '''limiting magnetic fields for the various ranges of
                    measurements''') 
    
    #MEASUREMENTS PARAMETERS
    parser.add_argument('--generatingMode', default = False, action = 'store_true',
                        help = 'activate if you want to generate a number of ground states and low energy states')
    parser.add_argument('--energy', default = False, action = 'store_true',
                        help = 'activate if you want to save the energy')
    parser.add_argument('--magnetisation', default = False, action = 'store_true',
                        help = 'activate if you want to save the magnetisation')
    parser.add_argument('--magnstats', default = False, action = 'store_true', 
                       help = 'activate if you want to compute the magnetisation statistics')
    parser.add_argument('--charges', default = False, action = 'store_true',
                        help = 'activate if you want to save the charges')
    parser.add_argument('--frustratedT', default = False, action = 'store_true',
                        help = 'activate if you want to save the frustrated triangles')
    parser.add_argument('--correlations', default = False, action = 'store_true',
                        help = 'activate if you want to save either central or all correlations')
    parser.add_argument('--both', default = False, action = 'store_true',
                        help = '''activate if you want to save both''')
    parser.add_argument('--firstcorrelations', default = False, action = 'store_true',
                        help = 'activate if you want to save first correlations, otherwise will save central')
    parser.add_argument('--sref0', nargs = '+', type = int, default = [], help = 'ref spin 0')
    parser.add_argument('--sref1', nargs = '+', type = int, default = [], help = 'ref spin 1')
    parser.add_argument('--sref2', nargs = '+', type = int, default = [], help = 'ref spin 2')
    #SAVE
    parser.add_argument('--output', type = str, default = "randomoutput.dat", help = 'saving filename (.pkl will be added)')
    parser.add_argument('--verbose', default = False, action = 'store_true', help = 'verbose')
    args = parser.parse_args()
    
    main(args)

