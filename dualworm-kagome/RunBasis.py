
# coding: utf-8

# In[ ]:

import numpy as np
import dimers as dim
import DualwormFunctions as dw
import RunBasisFunctions as rbf
import StartStates as strst
import Observables as obs

import pickle
from safe import safe

from time import time

import argparse


# In[ ]:

def main(args):
    
    print("-------------------Initialisation--------------------")
    ### PREPARE SAVING
    backup = safe()
    backup.params = safe()
    backup.results = safe()

    ### SIMULATIONS PARAMETERS
    
    loadfromfile = args.loadfromfile
    if loadfromfile:
        f = open("./" + args.filename +'.pkl','rb')
        loadbackup = pickle.load(f) #save the parameters
        f.close()
        L = loadbackup.params.L
        assert L == args.L, "Loaded and required lattice sizes not compatible."
    else:
        loadbackup = safe()
       
    backup.params.L = L = args.L
    print('Lattice side size: ', L)
    [d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, 
     d_nd, d_vd, d_wn, sidlist, didlist, c_ijl, ijl_c, c2s, csign] =\
    dw.latticeinit(L)
    
    [couplings,h, hamiltonian, ssf, alternate, s2p, temperatures,
     betas, nt] =\
    rbf.SimulationParameters(args,backup, d_ijl,
                             ijl_d, ijl_s, s_ijl, s2_d, L)
        
    ### SIMULATION INITIALISATION
    (states, energies, spinstates, ref_energies) =    rbf.StatesAndEnergyInit(args, backup, loadbackup,hamiltonian,
                        nt, d_ijl, d_2s, s_ijl,
                       couplings, h, L)
    
    backup.params.ncores = ncores = args.ncores
    
    if h == 0:
        rbf.CheckGs(args, ref_energies, energies)
    
    rbf.CheckEnergies(hamiltonian, energies, states, spinstates, h, nt, d_2s)
        
    
    ### INITIALISATION FOR THE MEASUREMENTS

    
    #Observables to measure
    [nnlists, observables, observableslist, magnfuncid,cfuncid]  =    rbf.ObservablesInit(args, backup, s_ijl, ijl_s, L)
    
    # Temperatures to measure
    stat_temps = rbf.Temperatures2MeasureInit(args, backup,
                                             temperatures, nt)

    ## THERMALISATION
    #preparation
    ncores = args.ncores
    
    
    print("-----------Thermalisation------------------")
    nb = 1 # only one bin, no statistics
    num_in_bin = args.nst# mcs iterations per bins
    iterworm = nips = args.nips #number of worm iterations before considering swaps
    nmaxiter = args.nmaxiter
    statsfunctions = [] #don't compute any statistics
    check = 0 #don't turn to spins to check
    print('Number of thermalisation steps = ', num_in_bin*nb)
    backup.params.thermsteps = num_in_bin*nb
    backup.params.ncores = ncores = args.ncores
    
    kw = {'nb':nb,'num_in_bin':num_in_bin, 'iterworm':iterworm,
          'nitermax':nmaxiter,'check':check,
          'statsfunctions':statsfunctions,
          'nt':nt, 'hamiltonian':hamiltonian,'ncores':ncores,
          'd_nd':d_nd,'d_vd':d_vd,'d_wn':d_wn, 'd_2s':d_2s, 's2_d':s2_d,
          'sidlist':sidlist,'didlist':didlist,'s_ijl':s_ijl,'ijl_s':ijl_s,
          'L':L, 'h':h, 's2p':s2p, 'ssf':ssf}


    t1 = time()
    (meanstatth, swapsth, failedupdatesth) = dw.mcs_swaps(states, spinstates, energies, betas, [], **kw)
    t2 = time()
    
    print('Time for all thermalisation steps = ', t2-t1)
    
    backup.results.swapsth = swapsth
    backup.results.failedupdatesth = failedupdatesth
    
    rbf.CheckEnergies(hamiltonian, energies, states, spinstates, h, nt, d_2s, False)
    
    if h == 0:
        rbf.CheckGs(args,ref_energies, energies)
    
    ## MEASUREMENT PREPARATION 

    print("-----------Measurements-----------------")
    # Preparation to call the method
    backup.params.nb = nb = args.nb # number of bins
    backup.params.num_in_bin = num_in_bin = args.nsm//nb
    print('Number of measurement steps = ', num_in_bin*nb) # number of iterations = nb * num_in_bin 
    iterworm = nips #number of worm iterations before considering swaps and measuring the state again
    statsfunctions = observables #TODO set functions
    backup.results.namefunctions = observableslist #TODO set functions corresponding to the above
    print(backup.results.namefunctions)
    check = 1 #turn to spins and check match works
    backup.params.measperiod = measperiod = args.measperiod
    print('Measurement period:', measperiod)
    backup.params.measupdate = measupdate = args.measupdate
    if measupdate:
        nnspins, s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)
        backup.params.p = p = args.p
    else:
        if not ssf:
            nnspins = []
            s2p = []
            p = 0
        else:
            nnspins = []
            p = 0
    
    backup.params.magnstats = magnstats = args.magnstats
            
    kw = {'nb':nb,'num_in_bin':num_in_bin, 'iterworm':iterworm,
          'nitermax':nmaxiter,'check':check,
          'statsfunctions':statsfunctions,
          'nt':nt, 'hamiltonian':hamiltonian,
          'nnlists':nnlists,
          'd_nd':d_nd,'d_vd':d_vd,'d_wn':d_wn, 'd_2s':d_2s, 's2_d':s2_d,
          'sidlist':sidlist,'didlist':didlist,'s_ijl':s_ijl,'ijl_s':ijl_s,'L':L,
          'ncores':ncores, 'measupdate': measupdate, 'nnspins': nnspins, 's2p':s2p, 
          'magnstats':magnstats,'magnfuncid':magnfuncid, 'p':p,
          'c2s':c2s, 'csign':csign, 'measperiod':measperiod, 'h':h,'ssf':ssf}
        
    # Run measurements
    t1 = time()
    (backup.results.meanstat, backup.results.swaps, backup.results.failedupdates) = (meanstat, swaps, failedupdates) = dw.mcs_swaps(states, spinstates, energies, betas, stat_temps,**kw)
    t2 = time()
    print('Time for all measurements steps = ', t2-t1)

    rbf.CheckEnergies(hamiltonian, energies, states, spinstates, h, nt, d_2s, True)
    
    if h == 0:
        rbf.CheckGs(args,ref_energies, energies)
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

    #Save the final results
    backup.results.t_meanfunc = t_meanfunc
    backup.results.t_varmeanfunc = t_varmeanfunc
    backup.results.states = states
    backup.results.spinstates = spinstates
    #Save the backup object in a file
    pickle.dump(backup, open(args.output + '.pkl','wb'))
    print("Job done")
    return meanstat, failedupdatesth, failedupdates


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
    parser.add_argument('--measperiod', type = int, default = 1,
                        help = 'number of nips worm building + swaps between measurements')
    parser.add_argument('--nb', type = int, default = 20,
                        help = 'number of bins')

    #PARALLELISATION
    parser.add_argument('--ncores', type = int, default = 4,
                        help = 'number of threads to use')

    #INITIALISATION PARAMETERS
    parser.add_argument('--randominit', default = False, action ='store_true',
                        help = 'intialise the states randomly')
    parser.add_argument('--same', default = False, action = 'store_true',
                        help = '''initialise all temperatures with the same
                        state (debug purposes)''')
    parser.add_argument('--magninit', default = False, action = 'store_true',
                        help = '''initialise all the temperature with the maximally magnetised GS''')
    parser.add_argument('--loadfromfile', default = False, action = 'store_true',
                       help = '''initialise all the states with
                       results from a previously performed simulations''')
    parser.add_argument('--filename', type = str, default = "", 
                        help = '''name of the previously performed simulation''')
    
    #WORM PARAMETERS
    parser.add_argument('--nmaxiter', type = int, default = 10,
                        help = '''maximal number of segments in a loop update over the
                        size of the lattice (1 = 1times the number of dualbonds in the
                        lattice)''')
    parser.add_argument('--measupdate', default = False, action = 'store_true',
                       help = '''activate to mimic the action of the measuring tip''')
    parser.add_argument('--p', type = float, default = 0.1, 
                       help = '''prob of the measuring tip flipping the spin (number between 0 and 1)''')
    parser.add_argument('--ssf', default = False, action = 'store_true',
                        help = 'activate for single spin flip update')
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

