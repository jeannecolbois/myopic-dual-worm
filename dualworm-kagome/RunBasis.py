
# coding: utf-8

# In[ ]:


import numpy as np
import dimers as dim
import DualwormFunctions as dw
import StartStates as strst
import Observables as obs

import pickle
from safe import safe

from time import time

import argparse


# In[ ]:


def main(args):
    ### PREPARE SAVING
    backup = safe()
    backup.params = safe()
    backup.results = safe()
    ### SIMULATIONS INITIATLISATION
    backup.params.loadfromfile = loadfromfile = args.loadfromfile
    if loadfromfile:
        f = open(args.filename +'.pkl', 'rb')
        loadbackup = pickle.load(f) #save the parameters
        L = loadbackup.params.L
        
        assert L == args.L, "Loaded and required lattice sizes not compatible."
    
        f.close
        
    backup.params.L = L = args.L
    print('Lattice side size: ', L)
    [d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, d_nd, d_vd, d_wn,
     sidlist, didlist, c_ijl, ijl_c, c2s, csign] = dw.latticeinit(L)


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
    couplings = {'J1': J1, 'J2':J2, 'J3':J3, 'J3st':J3st, 'J4':J4}
    print("Couplings extracted")
    hamiltonian = dw.Hamiltonian(couplings,d_ijl, ijl_d, L)
    print("Hamiltonian expression (without field) computed")


    backup.params.ssf = ssf = args.ssf
    backup.params.alternate = alternate = args.alternate

    s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)
    if ssf:
        print("single spin flip update")
    if alternate:
        print("alternating ssf and dw update")
    if ssf or alternate:
        nnspins, s2p = dw.spin2plaquette(ijl_s, s_ijl, s2_d,L)

    assert (not ((ssf or alternate) and (J2 != 0 or J3 !=0 or J3st != 0 or J4!=0))),    "The ssf is only available with J1"



    ## Temperatures and magnetic fields to simulate
    t_list = [t for t in args.t_list]
    nt_list = args.nt_list
    backup.params.loglist = loglist = args.log_tlist
    if loglist:
        temperatures = dw.create_log_temperatures(nt_list, t_list)
    else:
        temperatures = dw.create_temperatures(nt_list, t_list)
    betas = 1/temperatures
    backup.params.temperatures = temperatures
    # total number of different temperatures
    backup.params.nt = nt = len(temperatures)
    if args.h_list:
        h_list = [h for h in args.h_list]
        nh_list = args.nh_list
        hfields = dw.create_hfields(nh_list, h_list)
        nh = len(hfields)
    else:
        hfields = np.array([h])
        nh = 1
    backup.params.hfields = hfields;
    backup.params.nh = nh;

    print('Number of temperatures: ', nt)
    print('Temperatures:', temperatures)
    print('Number of magnetic fields: ', nh)
    print('Magnetic fields: ', hfields)

    [walker2params, walker2ids, ids2walker] = dw.walkerstable(betas, nt, hfields, nh)

    if loadfromfile:
        f = open('./' + args.filename +'.pkl', 'rb')
        loadbackup = pickle.load(f) #save the parameters
        
        states = loadbackup.results.states
        spinstates = loadbackup.results.spinstates
        energies = [[dw.compute_energy(hamiltonian, states[ids2walker[bid, hid]])
                  - hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                  for hid in range(nh)]
                 for bid in range(nt)]
    
        energies = np.array(energies)
        f.close()
    else:
        backup.params.randominit = randominit = args.randominit    
        print('Fully random initialisation = ', randominit)
        backup.params.same = same = args.same
        print('Identical initialisation = ', same)
        backup.params.magninit = magninit = args.magninit
        print('Magnetisation initialisation = ', magninit)
        backup.params.maxflip = maxflip = args.maxflip
        backup.params.magnstripes = magnstripes = args.magnstripes

        kwinit = {'random': randominit, 'same': same, 
                  'magninit': magninit, 'h':h, 'maxflip':maxflip,
                 'magnstripes': magnstripes}

        (states, energies,spinstates) = strst.statesinit(nt, nh, hfields,
                                        ids2walker,d_ijl, d_2s, s_ijl,
                                        hamiltonian, **kwinit)

    # end if
    
    #print("Energies = ", energies)
    print("Energies size: ", energies.shape)
    backup.params.ncores = ncores = args.ncores

    new_en_states = [[dim.hamiltonian(hamiltonian,
                                     states[ids2walker[bid,hid]])
                     -hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                     for hid in range(nh)] for bid in range(nt)]
    new_en_states = np.array(new_en_states)



    for t in range(nt):
        for h in range(nh):
            if np.absolute(energies[t,h]-new_en_states[t,h]) > 1.0e-12:
                print('RunBasis: Issue at temperature index ', t, ' and h index ', h)
                print("   energies[t,h] = ", energies[t,h])
                print("   H0[t,h] = ", dim.hamiltonian(hamiltonian,
                                                       states[ids2walker[t,h]]))
                print("   magntot[t,h] ", spinstates[ids2walker[t,h]].sum())
                print("   new_E[t,h] = H0[t,h] - h*magntot[t,h]",
                      dim.hamiltonian(hamiltonian, states[ids2walker[t,h]])
                      - hfields[h]*spinstates[ids2walker[t,h]].sum())
    ref_en_states = np.zeros(len(hfields))

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
        t = 0
        for h in range(nh):
            if np.absolute(ref_en_states[h]-new_en_states[t,h]) > 1.0e-12:
                print('RunBasis: Away from gs at t index ', t, ' and h index ', h)
                print("   new_en_states[t,h] = ", new_en_states[t,h])
                print("   ref_en_states[h] = ", ref_en_states[h])


    if not dw.statescheck(spinstates, states, d_2s):
        mistakes = [dw.onestatecheck(spinstate, state, d_2s) for spinstate, state in zip(spinstates, states)]
        print('Mistakes: ', mistakes)
    ### INITIALISATION FOR THE MEASUREMENTS

    # Observables to measure
    nnlists = []
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
        magnfuncid = observableslist.index('Magnetisation')
    else:
        magnfuncid = -1
    backup.params.charges = charges = args.charges
    if charges:
        observables.append(obs.charges)
        observableslist.append('Charges')
        cfuncid = observableslist.index('Magnetisation')
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
                print("Check: length of 3rdNN pairs list: ", len(dw.NN3pairs(ijl_s, s_ijl, L)))
                print("Check: length of 4thNN pairs list: ", len(dw.NN4pairs(ijl_s, s_ijl, L)))
                nnlists = [dw.NNpairs(ijl_s, s_ijl, L), dw.NN2pairs(ijl_s, s_ijl, L),
                           dw.NN3pairs(ijl_s, s_ijl, L), dw.NN4pairs(ijl_s, s_ijl, L)]
                observables.append(obs.firstcorrelations)
                observableslist.append('FirstCorrelations')
            else:
                observables.append(obs.centralcorrelations)
                observableslist.append('Central_Correlations')

    print('List of measurements to be performed:', observableslist)

    # Temperatures to measure
    if args.stat_temps_lims is None:
        #by default, we measure all the temperatures
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

    backup.params.stat_temps = stat_temps
    assert len(stat_temps) <= nt,    'The number of temperature indices to measure cannot be bigger than    the number of temperatures.'

    # magnetic fields to measure
    if args.stat_hfields_lims is None:
        #by default, we measure all the temperatures
        stat_hfields = range(nh)
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
    backup.params.stat_hfields = stat_hfields
    assert len(stat_hfields) <= nh,    'The number of field indices to measure cannot be bigger than    the number of fields.'

    ## THERMALISATION
    #preparation
    print("-----------Thermalisation------------------")
    nb = 1 # only one bin, no statistics
    num_in_bin = args.nst# mcs iterations per bins
    iterworm = nips = args.nips # maximal number of MCS iterations
    # before considering swaps (one iteration = one system size update)
    backup.params.nrps = nrps = args.nrps # number of replica loop iterations in a MC step
    nmaxiter = args.nmaxiter
    statsfunctions = [] #don't compute any statistics
    check = False #don't turn to spins to check
    print('Number of thermalisation steps = ', num_in_bin*nb)
    backup.params.thermsteps = num_in_bin*nb
    backup.params.ncores = ncores = args.ncores
    #launch thermalisation
    
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
    
    #print("Energies = ", energies)
    print("Energies size: ", energies.shape)
    backup.results.swapst_th = swapst_th
    backup.results.swapsh_th = swapsh_th
    backup.results.failedupdatesth = failedupdatesth
    print('Time for all thermalisation steps = ', t2-t1)
    new_en_states = [[dim.hamiltonian(hamiltonian,
                                     states[ids2walker[bid,hid]])
                     -hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                     for hid in range(nh)] for bid in range(nt)]
    new_en_states = np.array(new_en_states)
    for t in range(nt):
        for h in range(nh):
            if np.absolute(energies[t,h]-new_en_states[t,h]) > 1.0e-9:
                print('RunBasis: Issue at temperature index ', t, ' and h index ', h)
                print("   energies[t,h] = ", energies[t,h])
                print("   H0[t,h] = ", dim.hamiltonian(hamiltonian,
                                                       states[ids2walker[t,h]]))
                print("   magntot[t,h] ", spinstates[ids2walker[t,h]].sum())
                print("   new_E[t,h] = H0[t,h] - h*magntot[t,h]",
                      dim.hamiltonian(hamiltonian, states[ids2walker[t,h]])
                      - hfields[h]*spinstates[ids2walker[t,h]].sum())

    if args.checkgs:
        t = 0
        for h in range(nh):
            if np.absolute(ref_en_states[h]-new_en_states[t,h]) > 1.0e-12:
                print('RunBasis: Away from gs at t index ', t, ' and h index ', h)
                print("   new_en_states[t,h] = ", new_en_states[t,h])
                print("   ref_en_states[h] = ", ref_en_states[h])

    #print("walker2params shape: ", walker2params.shape)
    #print("walker2ids shape: ", walker2ids.shape)
    #print("ids2walker shape: ", ids2walker.shape)
    #
    #print("walker2params = ", walker2params)
    #print("walker2ids = ", walker2ids)
    #print("ids2walker = ", ids2walker)
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
    check = True #turn to spins and check match works
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
                
    kw = {'nb':nb,'num_in_bin':num_in_bin, 'iterworm':iterworm,
          'nrps': nrps,
          'nitermax':nmaxiter,'check':check,
          'statsfunctions':statsfunctions,
          'nt':nt, 'hamiltonian':hamiltonian,
          'nnlists':nnlists,
          'd_nd':d_nd,'d_vd':d_vd,'d_wn':d_wn, 'd_2s':d_2s, 's2_d':s2_d,
          'sidlist':sidlist,'didlist':didlist,'s_ijl':s_ijl,'ijl_s':ijl_s,'L':L,
          'ncores':ncores, 'measupdate': measupdate, 'nnspins': nnspins, 's2p':s2p,'magnfuncid':magnfuncid, 'p':p,
          'c2s':c2s, 'csign':csign, 'measperiod':measperiod,
          'nh':nh, 'hfields':hfields, 'walker2params':walker2params,
          'walker2ids':walker2ids,'ids2walker':ids2walker,
          'ssf':ssf, 'alternate':alternate, 'randspinupdate': False}
        # Run measurements
    t1 = time()
   
    (backup.results.statstable, backup.results.swapst, backup.results.swapsh,
     backup.results.failedupdates) =\
    (statstable, swapst, swapsh, failedupdates) =\
    dw.mcs_swaps(states, spinstates, energies, betas, stat_temps, stat_hfields,**kw)
    #print("Energies = ", energies)
    print("Energies size: ", energies.shape)
    t2 = time()
    print('Time for all measurements steps = ', t2-t1)
    new_en_states = [[dim.hamiltonian(hamiltonian,
                                     states[ids2walker[bid,hid]])
                     -hfields[hid]*spinstates[ids2walker[bid,hid]].sum()
                     for hid in range(nh)] for bid in range(nt)]
    new_en_states = np.array(new_en_states)
    for t in range(nt):
        for h in range(nh):
            if np.absolute(energies[t,h]-new_en_states[t,h]) > 1.0e-9:
                print('RunBasis: Issue at temperature index ', t, ' and h index ', h)
                print("   energies[t,h] = ", energies[t,h])
                print("   H0[t,h] = ", dim.hamiltonian(hamiltonian,
                                                       states[ids2walker[t,h]]))
                print("   magntot[t,h] ", spinstates[ids2walker[t,h]].sum())
                print("   new_E[t,h] = H0[t,h] - h*magntot[t,h]",
                      dim.hamiltonian(hamiltonian, states[ids2walker[t,h]])
                      - hfields[h]*spinstates[ids2walker[t,h]].sum())

    if args.checkgs:
        t = 0
        for h in range(nh):
            if np.absolute(ref_en_states[h]-new_en_states[t,h]) > 1.0e-12:
                print('RunBasis: Away from gs at t index ', t, ' and h index ', h)
                print("   new_en_states[t,h] = ", new_en_states[t,h])
                print("   ref_en_states[h] = ", ref_en_states[h])

    #print("-----------Computing statistics----------------")
    ### STATISTICS ##
    #t_h_meanfunc = list() #for each function, for each temperature,for each field, mean of the state function
    #t_h_varmeanfunc = list() #for each function, for each temperature,for each field, variance of the state function
    #numsites = len(s_ijl)

    #for idtuple, stattuple in enumerate(statstable):
    #    # means per magnetic field and temperature
    #    # note: stattuple[tupleindex][resid][reshid][bid] where bid is the bin index
    #    t_h_meanfunc.append((np.array(stattuple[0]).sum(2)/nb, np.array(stattuple[1]).sum(2)/nb))

    #    #variances:
    #    tuplevar1 = [[0 for h in stat_hfields] for t in stat_temps]
    #    tuplevar2 = [[0 for h in stat_hfields] for t in stat_temps]
    #    for resid, t in enumerate(stat_temps):
    #        for reshid, h in enumerate(stat_hfields):
    #            for b in range(nb):
    #                tuplevar1[resid][reshid] += ((stattuple[0][resid][reshid][b] - t_h_meanfunc[idtuple][0][resid][reshid]) ** 2)/(nb * (nb - 1))
    #                tuplevar2[resid][reshid] += ((stattuple[1][resid][reshid][b] - t_h_meanfunc[idtuple][1][resid][reshid]) ** 2)/(nb * (nb - 1))
    #    t_h_varmeanfunc.append((tuplevar1, tuplevar2))

    # Additional results for the correlations are handled directly in AnalysisBasis_3dot1dot5

    #backup.results.t_h_meanfunc = t_h_meanfunc
    #backup.results.t_h_varmeanfunc = t_h_varmeanfunc
    
    #Save the final results
    backup.results.states = states
    backup.results.spinstates = spinstates
    #Save the backup object in a file
    pickle.dump(backup, open(args.output + '.pkl','wb'))
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


# In[ ]:


#    t_meanfunc = list() #for each function, for each temperature, mean of the state function
#    t_varmeanfunc = list() #for each function, for each temperature, variance of the state function
#    numsites = len(s_ijl)
#    if not magnstats:
#        for idtuple, stattuple in enumerate(meanstat):
#            # means:
#            t_meanfunc.append((np.array(stattuple[0]).sum(1)/nb, np.array(stattuple[1]).sum(1)/nb))
#
#            #variances:
#            tuplevar1 = [0 for t in stat_temps]
#            tuplevar2 = [0 for t in stat_temps]
#            for resid, t in enumerate(stat_temps):
#                for b in range(nb):
#                    tuplevar1[resid] += ((stattuple[0][resid][b] - t_meanfunc[idtuple][0][resid]) ** 2)/(nb * (nb - 1))
#                    tuplevar2[resid] += ((stattuple[1][resid][b] - t_meanfunc[idtuple][1][resid]) ** 2)/(nb * (nb - 1))
#            t_varmeanfunc.append((tuplevar1, tuplevar2))
#
#    # Additional results for the correlations are handled directly in AnalysisBasis_3dot1dot5
#    print("avg magn: ", t_meanfunc[magnfuncid])
#    #Save the final results
#    backup.results.t_meanfunc = t_meanfunc
#    backup.results.t_varmeanfunc = t_varmeanfunc

