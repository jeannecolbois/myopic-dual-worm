
# coding: utf-8

# In[ ]:

import numpy as np
import dimers as dim
import DualwormFunctions as dw
import StartStates as strst
import Observables as obs
import RunBasisFunctions_NewSaves as rbf

import hickle as hkl
from safe import safe
import os

from time import time

import argparse


# In[ ]:

def main(args):
    
    print("-------------------Initialisation--------------------")
    ### PREPARE SAVING
    backup = rbf.SafePreparation(args)
    
    ### SIMULATIONS PARAMETERS
    loadfromfile = args.loadfromfile
    loadbackup = "./" + args.filename + ".hkl"
    if loadfromfile:
        L = hkl.load(loadbackup, path = "/parameters/L")
        assert L == args.L, "Loaded and required lattice sizes not compatible."
    
    L = args.L
    
    print('Lattice side size: ', L)
    [d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, d_nd, d_vd, d_wn,
     sidlist, didlist, c_ijl, ijl_c, c2s, csign] =\
    dw.latticeinit(L)
    
    [couplings, hamiltonian, ssf, alternate, s2p, temperatures,
     betas, nt, hfields, nh] =\
    rbf.SimulationParameters(args,backup, loadfromfile, d_ijl,
                             ijl_d, ijl_s, s_ijl, s2_d, L)
    
    [walker2params, walker2ids, ids2walker] =    dw.walkerstable(betas, nt, hfields, nh)

    # Saving the status:
    
    ## SIMULATION INITIALISATION
    (states, energies, spinstates, ref_en_states) =    rbf.StatesAndEnergyInit(args, backup, loadbackup,hamiltonian,
                            ids2walker, nt, nh, hfields, d_ijl,
                            d_2s, s_ijl, couplings, L)
    
    # Check g.s. (if required)
    rbf.CheckGs(args, ref_en_states, energies, nh)

    # Check states consistency:
    if not dw.statescheck(spinstates, states, d_2s):
        mistakes = [dw.onestatecheck(spinstate, state, d_2s) for spinstate, state in zip(spinstates, states)]
        print('Mistakes: ', mistakes)
    
    ### INITIALISATION FOR THE MEASUREMENTS
    # Observables to measure
    [nnlists, observables, observableslist, magnfuncid,cfuncid]  =    rbf.ObservablesInit(args, backup, s_ijl, ijl_s, L)
    
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
    hkl.dump(thermalisation,backup,
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
          's2p':s2p, 'ssf':ssf, 'alternate':alternate}


    t1 = time()
    (statstableth, swapst_th, swapsh_th, failedupdatesth) =     dw.mcs_swaps(states, spinstates, energies, betas, [],[], **kw)
    t2 = time()
    
    print('Time for all thermalisation steps = ', t2-t1)
    
    
    thermres = {'swapst_th':swapst_th, 'swapsh_th':swapsh_th,
               'failedupdatesth':failedupdatesth, 'totaltime':t2-t1}
    hkl.dump(thermres,backup,
             path = "/results/thermres", mode = 'r+')
    
    
    rbf.CheckEnergies(hamiltonian, energies, states, spinstates,
                      hfields, nh, ids2walker, nt)

    rbf.CheckGs(args,ref_en_states, energies, nh)
    
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
    measupdate = args.measupdate
    if measupdate:
        nnspins, s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)
        p = args.p
    else:
        if not ssf:
            nnspins = []
            s2p = []
            p = 0
        else:
            nnspins = []
            p = 0
    
    kwmeas = {'nb':nb, 'num_in_bin':num_in_bin,'nips':nips,
             'measperiod':measperiod, 'measupdate':measupdate,
             'nnspins':nnspins, 's2p': s2p}
    hkl.dump(kwmeas, backup, path = "/parameters/measurements", mode = 'r+')
    kw = {'nb':nb,'num_in_bin':num_in_bin, 'iterworm':iterworm,
          'nrps': nrps,
          'nitermax':nmaxiter,'check':check,
          'statsfunctions':statsfunctions,
          'nt':nt, 'hamiltonian':hamiltonian,
          'nnlists':nnlists,
          'd_nd':d_nd,'d_vd':d_vd,'d_wn':d_wn, 'd_2s':d_2s, 's2_d':s2_d,
          'sidlist':sidlist,'didlist':didlist,'s_ijl':s_ijl,'ijl_s':ijl_s,'L':L,
          'ncores':ncores, 'measupdate': measupdate, 'nnspins': nnspins, 's2p':s2p,
          'magnfuncid':magnfuncid, 'p':p,
          'c2s':c2s, 'csign':csign, 'measperiod':measperiod,
          'nh':nh, 'hfields':hfields, 'walker2params':walker2params,
          'walker2ids':walker2ids,'ids2walker':ids2walker,
          'ssf':ssf, 'alternate':alternate, 'randspinupdate': False}
        # Run measurements
    
    t1 = time()
    (statstable, swapst, swapsh, failedupdates) =    dw.mcs_swaps(states, spinstates, energies, betas, stat_temps, stat_hfields,**kw)
    #print("Energies = ", energies)
    t2 = time()
    
    print('Time for all measurements steps = ', t2-t1)
    print("Energies size: ", energies.shape)
    
    measurementsres = {'swapst': swapst, 'swapsh': swapsh,
                       'failedupdates':failedupdates}
    
    hkl.dump(measurementsres, backup, path = "/results/measurements", mode = 'r+')
    
    hkl.dump(statstable, backup+"_statstable")
    
    new_en_states = [[dim.hamiltonian(hamiltonian,
                                     states[ids2walker[bid,hid]])
                     -hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                     for hid in range(nh)] for bid in range(nt)]
    
    rbf.CheckEnergies(hamiltonian, energies, states, spinstates,
                      hfields, nh, ids2walker, nt)

    rbf.CheckGs(args,ref_en_states, energies, nh)
                
    #Save the final results
    hkl.dump(states, backup+"_states")
    hkl.dump(spinstates, backup+"_spinstates")
    
    print("Job done")
    return statstable, swapst, swapsh, failedupdatesth, failedupdates


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
    parser.add_argument('--J4', type = float, default = 0.0,
                        help = '4th NN coupling')
    parser.add_argument('--h', type = float, default = 0.0,
                        help = 'Magnetic field')
    
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
    parser.add_argument('--filename', type = str, help = '''initialise all the states with
                       results from a previously performed simulations''')
    
    #WORM PARAMETERS
    parser.add_argument('--nmaxiter', type = int, default = 10,
                        help = '''maximal number of segments in a loop update over the
                        size of the lattice (1 = 1times the number of dualbonds in the
                        lattice)''')
    parser.add_argument('--measupdate', default = False, action = 'store_true',
                       help = '''activate to mimic the action of the measuring tip''')
    parser.add_argument('--p', type = float, default = 0.0, 
                       help = '''prob of the measuring tip flipping the spin (number between 0 and 1)''')
    parser.add_argument('--ssf', default = False, action = 'store_true',
                        help = 'activate for single spin flip update')
    parser.add_argument('--alternate', default = False, action = 'store_true',
                        help = 'activate for single spin flip update and dw update')
    parser.add_argument('--checkgs', default = False, action = 'store_true',
                        help = 'activate to debug ssf')
    
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
    parser.add_argument('--h_list', nargs = '+', type = float,
                        help = 'list of limiting magnetic field values')
    parser.add_argument('--nh_list', nargs = '+', type = int, 
                        help = 'list of number of magnetic fields in between the given limiting temperatures')
    parser.add_argument('--stat_hfields_lims', nargs = '+', type = float,
                    help = '''limiting magnetic fields for the various ranges of
                    measurements''') 
    #CORRELATIONS PARAMETER
    parser.add_argument('--energy', default = False, action = 'store_true',
                        help = 'activate if you want to save the energy')
    parser.add_argument('--magnetisation', default = False, action = 'store_true',
                        help = 'activate if you want to save the magnetisation')
    parser.add_argument('--magnstats', default = False, action = 'store_true', 
                       help = 'activate if you want to compute the magnetisation statistics')
    parser.add_argument('--charges', default = False, action = 'store_true',
                        help = 'activate if you want to save the charges')
    parser.add_argument('--correlations', default = False, action = 'store_true',
                        help = 'activate if you want to save either central or all correlations')
    parser.add_argument('--all_correlations', default = False, action = 'store_true',
                        help = '''activate if you want to save the correlations for all non-equivalent
                        pairs of sites. Otherwise, will save central correlations.''')
    parser.add_argument('--firstcorrelations', default = False, action = 'store_true',
                        help = 'activate if you want to save first correlations, otherwise will save central')
    #SAVE
    parser.add_argument('--output', type = str, default = "randomoutput.dat", help = 'saving filename (.pkl will be added)')
    args = parser.parse_args()
    
    main(args)

