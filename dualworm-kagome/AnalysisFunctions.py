#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hickle as hkl
import KagomeFunctions as kf # "library" allowing to work on Kagome
import DualwormFunctions as dw
import KagomeDrawing as kdraw
import KagomeFT as kft
import Observables as obs
import warnings
import os
import itertools

# spline
from scipy.interpolate import InterpolatedUnivariateSpline


# In[ ]:


def correlationsTester(state, latsize, d_ijl, ijl_d, L):
    # J1 #
    couplings = {'J1':1.0}
    hamiltonian = dw.Hamiltonian(couplings, d_ijl, ijl_d, L)
    EJ1 = dw.compute_energy(hamiltonian, state, latsize)
    
    # J2 #
    couplings = {'J1':0, 'J2' : 1.0}
    hamiltonian = dw.Hamiltonian(couplings, d_ijl, ijl_d, L)
    EJ2 = dw.compute_energy(hamiltonian, state, latsize)
    
    # J3 #
    couplings = {'J1':0, 'J2' : 0, 'J3' : 1.0, 'J3st':0}
    hamiltonian = dw.Hamiltonian(couplings, d_ijl, ijl_d, L)
    EJ3 = dw.compute_energy(hamiltonian, state, latsize)
    
    # J3* #
    couplings = {'J1':0, 'J2' : 0, 'J3' : 0, 'J3st':1.0}
    hamiltonian = dw.Hamiltonian(couplings, d_ijl, ijl_d, L)
    EJ3st = dw.compute_energy(hamiltonian, state, latsize)
    
    
    # J4 #
    couplings = {'J1':0, 'J2' : 0, 'J3' : 0, 'J3st':0, 'J4': 1.0}
    hamiltonian = dw.Hamiltonian(couplings, d_ijl, ijl_d, L)
    EJ4 = dw.compute_energy(hamiltonian, state, latsize)
    
    config = {'J1':EJ1, 'J2': EJ2, 'J3':EJ3, 'J3st': EJ3st, 'J4': EJ4}
    
    return config


# In[ ]:


def LoadParameters(foldername, filenamelist, **kwargs):
    
    n = len(filenamelist)
    
    L = [[] for _ in range(n)]
    numsites = [0 for _ in range(n)]
    J1 = [0 for _ in range(n)]
    J2 = [0 for _ in range(n)]
    J3 = [0 for _ in range(n)]
    J3st = [0 for _ in range(n)]
    J4 = [0 for _ in range(n)]
    
    nb = [0 for _ in range(n)]
    num_in_bin = [0 for _ in range(n)]
    htip = [0 for _ in range(n)]
    Ttip = [0 for _ in range(n)]
    pswitch = [0 for _ in range(n)]
    uponly = [0 for _ in range(n)]
    path = [0 for _ in range(n)]
    
    temperatures = [[] for _ in range(n)]
    nt = [0 for _ in range(n)]
    stat_temps = [[] for _ in range(n)]
    temperatures_plots = [[] for _ in range(n)]
    
    
    hfields = [[] for _ in range(n)]
    nh = [0 for _ in range(n)]
    stat_hfields = [[] for _ in range(n)]
    hfields_plots = [[] for _ in range(n)]
    
    listfunctions = [[] for _ in range(n)]
    
    sref = [[] for _ in range(n)]
    ids2walker = [0 for _ in range(n)]
    
        
    merge = kwargs.get('mergeruns', False)
    group = kwargs.get('groupruns', 1)
    if merge:
        okformerge = True
    if group > 1:
        okforgroup = True
    for nf, filename in enumerate(filenamelist):
        [L[nf], numsites[nf], J1[nf], J2[nf], J3[nf], J3st[nf], J4[nf], nb[nf], 
         num_in_bin[nf], htip[nf], Ttip[nf], pswitch[nf],uponly[nf], path[nf],
         temperatures[nf], nt[nf], stat_temps[nf], temperatures_plots[nf],
         hfields[nf], nh[nf], stat_hfields[nf], hfields_plots[nf],
         listfunctions[nf], sref[nf], ids2walker[nf]] = LoadParametersFromFile(foldername, filename)
        if merge: # check that it makes sense to merge
            test = (L[nf] == L[0] and
                    J1[nf] == J1[0] and J2[nf] == J2[0] and 
                    J3[nf] == J3[0] and J3st[nf] == J3st[0] and 
                    J4[nf] == J4[0] and 
                    htip[nf] == htip[0] and Ttip[nf] == Ttip[0] and
                    pswitch[nf] == pswitch[0] and uponly[nf] == uponly[0] and
                    temperatures[nf] == temperatures[0]);
            if not test:
                okformerge = False
                raise Exception(" You required merging the runs but run 0 and " + nf + "are not compatible")
        if group >0  and nf > 0 and nf%group != 0:
            test = (L[nf] == L[nf-1] and
                    J1[nf] == J1[nf-1] and J2[nf] == J2[nf-1] and 
                    J3[nf] == J3[nf-1] and J3st[nf] == J3st[nf-1] and 
                    J4[nf] == J4[nf-1] and 
                    htip[nf] == htip[nf-1] and Ttip[nf] == Ttip[nf-1] and
                    pswitch[nf] == pswitch[nf-1] and uponly[nf] == uponly[nf-1] and
                    temperatures[nf] == temperatures[nf-1]);
            if not test:
                okforgroup = False
                raise Exception(" You required grouping the runs but runs "+ nf-1 +" and " + nf + "are not compatible")
           
    return L, numsites, J1, J2, J3, J3st, J4, nb, num_in_bin,             htip, Ttip, pswitch, uponly, path,             temperatures, nt,             stat_temps, temperatures_plots, hfields, nh,             stat_hfields, hfields_plots, listfunctions, sref, ids2walker


# In[ ]:


def LoadParametersFromFile(foldername, filename):
    backup = "./"+foldername+filename+".hkl"
    L = hkl.load(backup, path="/parameters/L")
    
    #spin table and dictionary
    (s_ijl, ijl_s) = kf.createspinsitetable(L)
    numsites = len(s_ijl)
    
    #couplings
    couplings = hkl.load(backup, path="/parameters/couplings")
    
    J1 = couplings['J1']
    J2 = couplings['J2']
    J3 = couplings['J3']
    J3st = couplings['J3st']
    J4 = couplings['J4']
    
    kwmeas = hkl.load(backup,  path = "/parameters/measurements")
    nb = kwmeas['nb']
    num_in_bin = kwmeas['num_in_bin']
    try:
        htip = kwmeas['htip']
        Ttip = kwmeas['Ttip']
        pswitch = kwmeas['pswitch']
        uponly = kwmeas['uponly']
        path = kwmeas['measupdatev']
    except:
        htip = 0
        Ttip = 0
        pswitch = 0
        uponly = True
        path = 1

    physical = hkl.load(backup, path = "/parameters/physical")
    temperatures = physical['temperatures'].tolist()
    nt = physical['nt']
    
    stat_temps = hkl.load(backup, path = "/parameters/stat_temps")
    temperatures_plots = [temperatures[t] for t in stat_temps]
    
    hfields = physical['hfields'].tolist()
    nh = physical['nh']
    
    stat_hfields = hkl.load(backup, path = "/parameters/stat_hfields")
    hfields_plots = [hfields[h] for h in stat_hfields]
    
    obsparams = hkl.load(backup, path = "/parameters/obsparams")
    listfunctions = obsparams['observableslist']
    
    try:
        srefs = obsparams['srefs']
    except:
        srefs = [ijl_s[(L, L, 0)], ijl_s[(L, L, 1)], ijl_s[(L, L, 2)]]
        print("srefs not registered.")
    if os.path.isfile("./"+foldername+filename+"_ids2walker.hkl"):
        ids2walker = hkl.load("./"+foldername+filename+"_ids2walker.hkl")
    else:
        ids2walker = []
        warnings.warn("ids2walker not found, not loaded!")
        

    return L, numsites, J1, J2, J3, J3st, J4, nb, num_in_bin,             htip, Ttip, pswitch, uponly, path,             temperatures, nt,             stat_temps, temperatures_plots, hfields, nh,             stat_hfields, hfields_plots, listfunctions, srefs, ids2walker


# In[ ]:


def ExtractStatistics(backup, idfunc, name,
                      nb, stat_temps, stat_hfields, sq = 0, **kwargs):
    '''
        This function gets the statistics from a file and
        computes the expectation values and variances of 
        the operator corresponding to idfunc
        
        sq = 0 -> not square stats
        sq = 1 -> square stats
    '''
    nb_drop = kwargs.get('nb_drop', 0)
    
    
    nb_stattuple = hkl.load(backup+"_"+name+"_final.hkl")
    
    
    t_h_meanfunc = nb_stattuple[nb_drop:nb].sum(0)/(nb-nb_drop)
    t_h_meanfunc = t_h_meanfunc[sq]
    
    
    binning = kwargs.get('binning', False)

    t_h_varmeanfunc = np.zeros(t_h_meanfunc.shape)
    #t_h_varmeanfunc_control = np.zeros(t_h_meanfunc.shape)
    
    t_h_varmeanfunc =  ((nb_stattuple[nb_drop:nb,sq,:,:] - t_h_meanfunc[np.newaxis,:,:])  ** 2).sum(0)/((nb -nb_drop)* (nb -nb_drop - 1))
        
    #for resid, t in enumerate(stat_temps):
    #    for reshid, h in enumerate(stat_hfields):
    #        for b in range(nb_drop, nb):
    #            t_h_varmeanfunc_control[resid,reshid] += ((nb_stattuple[b, sq, resid, reshid] - t_h_meanfunc[resid][reshid]) ** 2)
    #            # note that this is like t_h_varmeanfunc[resid,reshid, :]
    #        t_h_varmeanfunc_control[resid, reshid] = t_h_varmeanfunc_control[resid, reshid]/((nb -nb_drop)* (nb -nb_drop - 1))
        
    if binning:
        print("Binning..." + name)
        t_h_varmeanfunc = Binning(t_h_meanfunc,t_h_varmeanfunc, nb_stattuple[nb_drop:,sq,:,:], nb-nb_drop,
                                stat_temps, stat_hfields, name = name, sq = sq, **kwargs)
        
        if name == "FirstCorrelations":
            print(t_h_varmeanfunc[0][0])
    return t_h_meanfunc, t_h_varmeanfunc


# In[ ]:


def Binning(t_h_mean, t_h_varmean, stattuple, nb, stat_temps,stat_hfields, name = "", sq = 0, **kwargs):
    '''
        This function implements a binning analysis
    '''
    #raise Exception("Binning not adapted to the new statstable structure!")
    minbin = kwargs.get('minbin', 15)
    plzplot = kwargs.get('plzplot', False)
    plotmin = kwargs.get('plotmin', 0)
    plotmax = kwargs.get('plotmax', 10)
    plothmin = kwargs.get('plothmin', 0)
    plothmax = kwargs.get('plothmax', 10)
    
    minplt = max(0, plotmin)
    maxplt = min(plotmax, len(stat_temps))
    minhplt = max(0, plothmin)
    maxhplt = min(plothmax, len(stat_hfields))
    ### NAIVE IMPLEMENTATION:
    # go through all the measurements and recompute the variance when the bins are grouped
    
    # 1- preparing the list of bins
    nblist = []
    nbb = nb
    while nbb >= minbin:
        nblist.append(nbb)
        nbb = nbb//2
        
    print(" bins list for binning: ", nblist)
        
    # preparing the resulting variances
    t_h_vars = np.zeros(list(itertools.chain(*[[len(nblist)],t_h_varmean.shape])))#[[[] for reshid in range(len(stat_hfields))] for resid in range(len(stat_temps))]
    
    # go through the measurements
    #for resid, t in enumerate(stat_temps):
    #    for reshid, h in enumerate(stat_hfields):
    #        # go through the levels of binning
    #        for l,nbb in enumerate(nblist): 
    #            bin_avg = np.array(stattuple[0:(2**l),resid,reshid]).sum(0)/(2**l) # average the first 2**l bins
    #            asqbar = bin_avg**2/nbb
    #            t_h_vars[l,resid,reshid]=((bin_avg - t_h_mean[resid,reshid])**2)/(nbb*(nbb-1))
    #            for b in range(1,nbb):
    #                bin_avg = np.array(stattuple[(2**l)*b:(2**l)*(b+1),resid,reshid]).sum(0)/(2**l) # average the current
    #                #2**l bins
    #                asqbar += bin_avg**2/nbb
    #                t_h_vars[l,resid,reshid]+=((bin_avg - t_h_mean[resid,reshid])**2)/(nbb*(nbb-1))
    #                
    #            otherError=((asqbar - t_h_mean[resid,reshid]**2)/(nbb-1))
    #            if resid == minplt:
    #                print(nbb, " --- bin_avg: ", bin_avg, " --- error: (var)", t_h_vars[l,resid,reshid])#, " -- other: ", otherError)
    #            t_h_vars[l,resid,reshid] = max(abs(t_h_vars[l,resid,reshid]), abs(otherError))
    
    #print("NEW IMPLEMENTATION")
    print(stattuple.shape)
    
    t_h_vars_opti = np.zeros(list(itertools.chain(*[[len(nblist)],t_h_varmean.shape])))#[[[] for reshid in range(len(stat_hfields))] for resid in range(len(stat_temps))]
    for l,nbb in enumerate(nblist): 
        bin_avg = np.array([stattuple[(2**l)*b:(2**l)*(b+1),:,:].sum(0)/(2**l) for b in range(nbb)])# average the first 2**l bins
        print(bin_avg.shape)
        asqbar = np.mean(bin_avg**2, axis = 0) # still elementwise, shape 
        
        t_h_vars_opti[l,:,:] = (asqbar - t_h_mean**2)/(nbb-1)
        if np.all(t_h_vars_opti > -1e-15):
            t_h_vars_opti[l,:,:] = np.abs(t_h_vars_opti[l,:,:])
    if plzplot:
        print('plotting!')
        plt.figure(figsize=(12, 8),dpi=300)
        
        print(" minhplt: ", minhplt)
        print(" maxhplt: ", maxhplt)
        print(" minplt: ", minplt)
        print(" maxplt: ", maxplt)
        for reshid, h in enumerate(stat_hfields[minhplt:maxhplt]):
            for resid, t in enumerate(stat_temps[minplt:maxplt]):
                if len(t_h_vars.shape) ==3:
                    plt.plot(range(len(t_h_vars_opti[:,resid+minplt,reshid])), np.sqrt(t_h_vars_opti[:,resid+minplt,reshid+minhplt]), '.', label = 'tid = {0}'.format(t))
                else:
                    plt.plot(range(len(t_h_vars_opti[:,resid+minplt,reshid,0])), np.sqrt(t_h_vars_opti[:,resid+minplt,reshid+minhplt,0]), '.', label = 'tid = {0}'.format(t))
        #plt.title('h = {0}'.format(h))
        plt.xlabel('Binning level')
        if sq == 0:
            plt.ylabel(r'$\Delta$'+name)
        else:
            plt.ylabel(r'$\Delta$'+name+r"$^2$")
        plt.ylim([0, None])
        plt.grid(which='both')
        plt.legend()
        plt.show()
    
    # taking the max over l (i.e. on axis 0)
    t_h_varmean = np.amax(t_h_vars_opti,0)
    return np.array(t_h_varmean)


# In[ ]:


def LoadSwaps(foldername, filenamelist, nb, num_in_bin, nh, nt, **kwargs):
    n = len(filenamelist)
    swapst_th = [[] for _ in range(n)]
    swapsh_th = [[] for _ in range(n)]
    swapst = [[] for _ in range(n)]
    swapsh = [[] for _ in range(n)]
    
    for nf, filename in enumerate(filenamelist):
        [swapst_th[nf], swapsh_th[nf],
        swapst[nf], swapsh[nf]] =\
        LoadSwapsFromFile(foldername, filename, nb[nf],
                          num_in_bin[nf], nh[nf], nt[nf])
    
    return swapst_th, swapsh_th, swapst, swapsh


# In[ ]:


def LoadSwapsFromFile(foldername, filename, nb, num_in_bin, nh, nt):
    backup = "./"+foldername+filename+".hkl"
    
    kwmeas = hkl.load(backup, path="/parameters/measurements")
    
    nrps = kwmeas['nrps']
    nsms = nb*num_in_bin
    measperiod = kwmeas['measperiod']
    
    thermres = hkl.load(backup, path="/results/thermres")
    swapst_th = thermres['swapst_th']
    swapsh_th = thermres['swapsh_th']
    
    meas = hkl.load(backup, path = "/results/measurements")
    swapst = meas['swapst']
    swapsh = meas['swapsh']
    swapst = 4*np.array(swapst)/(nsms*nrps*measperiod*nh)
    swapsh = 4*np.array(swapsh)/(nsms*nrps*measperiod*nt)
    
    return swapst_th, swapsh_th, swapst, swapsh


# In[ ]:


def LoadUpdates(foldername, filenamelist, nb, num_in_bin, size, **kwargs):
    n = len(filenamelist)
    failedupdates_th = [[] for _ in range(n)]
    failedssfupdates_th = [[] for _ in range(n)]
    failedupdates = [[] for _ in range(n)]
    failedssfupdates = [[] for _ in range(n)]
    
    for nf, filename in enumerate(filenamelist):
        [failedupdates_th[nf], failedssfupdates_th[nf],
        failedupdates[nf], failedssfupdates[nf]] =\
        LoadUpdatesFromFile(foldername, filename, nb[nf],
                          num_in_bin[nf], size[nf])
    
    return failedupdates_th, failedssfupdates_th, failedupdates, failedssfupdates


# In[ ]:


def LoadUpdatesFromFile(foldername, filename, nb, num_in_bin, size):
    backup = "./"+foldername+filename+".hkl"
    
    kwtherm = hkl.load(backup, path="/parameters/thermalisation")
    kwmeas = hkl.load(backup, path="/parameters/measurements")
    
    nips = kwmeas['nips']
    nsms = nb*num_in_bin
    measperiod = kwmeas['measperiod']
    
    nstepstherm = kwtherm['thermsteps']*nips
    nsteps = measperiod*nips*nsms
        
    thermres = hkl.load(backup, path="/results/thermres")
    failedupdates_th = thermres['failedupdatesth']
    failedssfupdates_th = thermres['failedssfupdatesth']
    
    failedupdates_th = failedupdates_th/nstepstherm
    failedssfupdates_th = failedssfupdates_th/(nstepstherm*size)
    
    meas = hkl.load(backup, path = "/results/measurements")
    failedupdates = meas['failedupdates']
    failedssfupdates = meas['failedssfupdates']
    #print(failedssfupdates)
    failedupdates = failedupdates/(measperiod*nsms)
    failedssfupdates = failedssfupdates/(nsteps*size)
    
    
    return failedupdates_th, failedssfupdates_th, failedupdates, failedssfupdates


# In[ ]:


def LoadUpdateLists(foldername, filenamelist):
    n = len(filenamelist)
    updatelists = [[] for _ in range(n)]
    
    for nf, filename in enumerate(filenamelist):
        updatelists[nf] =        LoadUpdateListsFromFile(foldername, filename)
    
    return updatelists


# In[ ]:


def LoadUpdateListsFromFile(foldername, filename):
    backup = "./"+foldername+filename+".hkl"
    
    kwtherm = hkl.load(backup, path="/parameters/thermalisation")
    kwmeas = hkl.load(backup, path="/parameters/measurements")
    
    meas = hkl.load(backup, path = "/results/measurements")
    updatelists = meas['updatelists']
       
    return updatelists


# In[ ]:


def LoadStates(foldername, filenamelist,L,nh, **kwargs):
    n = len(filenamelist)
    
    t_spinstates = [[] for _ in range(n)]
    t_states = [[] for _ in range(n)]
    t_charges = [[] for _ in range(n)]
    for nf, filename in enumerate(filenamelist):
        [t_spinstates[nf], t_states[nf], t_charges[nf]] =                 LoadStatesFromFile(foldername, filename, L[nf],nh[nf],**kwargs)
        
    return t_spinstates, t_states, t_charges


# In[ ]:


def LoadStatesFromFile(foldername, filename, L, nh, **kwargs):
    
    [d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, d_nd, d_vd, d_wn,
     sidlist, didlist, c_ijl, ijl_c, c2s, csign] =\
    dw.latticeinit(L)
    
    backup = "./"+foldername+filename
    
    t_h_spinstates = hkl.load(backup+"_spinstates.hkl")
    t_h_states = hkl.load(backup+"_states.hkl")
    t_h_charges = []

    #if nh == 1:
    #    t_h_charges = np.array([obs.charges(len(s_ijl),[],[],
    #                                         spinstate, s_ijl, ijl_s,c2s,
    #                                         csign)
    #                             for spinstate in t_h_spinstates])
    #else:
    #    t_h_charges = np.array([[obs.charges(len(s_ijl),[],[],
    #                                         spinstate, s_ijl, ijl_s,c2s,
    #                                         csign)
    #                             for spinstate in h_spinstates]
    #                            for h_spinstates in t_h_spinstates])
    #    
    return t_h_spinstates, t_h_states, t_h_charges


# In[ ]:


def LoadGroundStates(foldername, filenamelist,L,nh,iters, **kwargs):
    n = len(filenamelist)
    
    t_spinstates = [[] for _ in range(n)]
    t_states = [[] for _ in range(n)]
    t_charges = [[] for _ in range(n)]
    for nf, filename in enumerate(filenamelist):
        [t_spinstates[nf], t_states[nf], t_charges[nf]] =                 LoadGroundStatesFromFile(foldername, filename, L[nf],nh[nf],iters[nf],**kwargs)
        
    return t_spinstates, t_states, t_charges


# In[ ]:


def LoadGroundStatesFromFile(foldername, filename, L, nh,iters, **kwargs):
    
    [d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, d_nd, d_vd, d_wn,
     sidlist, didlist, c_ijl, ijl_c, c2s, csign] =\
    dw.latticeinit(L)
    
    backup = "./"+foldername+filename
    
    it_spinstates = []
    it_states = []
    it_charges = []
    for it in range(iters+1):
        groundspinstate = hkl.load(backup+"_groundspinstate_it{0}.hkl".format(it))
        groundstate = hkl.load(backup+"_groundstate_it{0}.hkl".format(it))
        if nh == 1:
            charges = obs.charges(len(s_ijl),[],[],
                                      groundspinstate, s_ijl, ijl_s,c2s,
                                      csign)
        else:
            charges = obs.charges(len(s_ijl),[],[],
                                  groundspinstate, s_ijl, ijl_s,c2s,
                                  csign)
        it_spinstates.append(groundspinstate)
        it_states.append(groundstate)
        it_charges.append(charges)
        
    it_spinstates = np.array(it_spinstates, dtype = 'int8')
    it_states = np.array(it_states, dtype = 'int8')
    it_charges = np.array(it_charges, dtype = 'int8')
    
    return it_spinstates, it_states, it_charges 


# In[ ]:


def ComputeEntropy(C, ErrC, temperatures, stat_hfields, nt, nh,**kwargs):
    RS = kwargs.get('RS', False)
    S0 = kwargs.get('S0', np.log(2))
    #nS = kwargs.get('nS', 2000)
    ts = np.array(temperatures)
    
    DeltaSmin = np.zeros((nt, nh))
    DeltaS = np.zeros((nt, nh))
    DeltaSmax = np.zeros((nt, nh))

    Carray = np.array(C)
    CoverT = np.copy(Carray)
    CminoverT = np.copy(np.array(C - ErrC))
    CmaxoverT = np.copy(np.array(C + ErrC))
    #for tid in range(nt):
    CminoverT= CminoverT/ts[:,np.newaxis] # newaxis is for the magnetic field
    CoverT= Carray/ts[:,np.newaxis]
    CmaxoverT= CmaxoverT/ts[:,np.newaxis]
    
    print("CE: CoverT shape ", CoverT.shape) 
    # integrate and build S
    for hid, h in enumerate(stat_hfields):
        # fit
        splmin = InterpolatedUnivariateSpline(ts, CminoverT[:,hid])
        spl = InterpolatedUnivariateSpline(ts, CoverT[:,hid])
        splmax = InterpolatedUnivariateSpline(ts, CmaxoverT[:,hid])

        for tid in range(nt):
            DeltaSmin[tid,hid] =            splmin.integral(ts[tid], ts[-1])
            DeltaS[tid,hid] =            spl.integral(ts[tid], ts[-1])
            DeltaSmax[tid,hid] =            splmax.integral(ts[tid], ts[-1])
            
    t_h_Smin = S0*np.ones((nt, nh)) - DeltaSmax;
    t_h_S = S0*np.ones((nt, nh)) - DeltaS;
    t_h_Smax = S0*np.ones((nt, nh)) - DeltaSmin;

    return t_h_S, t_h_Smin, t_h_Smax


# In[ ]:


def ComputeEntropyFromE(t_h_MeanE, t_h_varMeanE, temperatures, stat_hfields, nt, nh, EGS = 0, **kwargs):
    #RS = kwargs.get('RS', False)
    S0 = kwargs.get('S0', np.log(2))

    ts = np.array(temperatures)[::-1]
    betas = 1/ts;
    
    
    DeltaSmin = np.zeros((nt, nh))
    DeltaS = np.zeros((nt, nh))
    DeltaSmax = np.zeros((nt, nh))

    for hid, h in enumerate(stat_hfields):
        # fit
        DEmin = t_h_MeanE[::-1,hid]-2*np.sqrt(t_h_varMeanE[::-1,hid])-EGS;
        DE = t_h_MeanE[::-1,hid]-EGS;
        DEmax = t_h_MeanE[::-1,hid]+2*np.sqrt(t_h_varMeanE[::-1,hid])-EGS;
        
        splmin = InterpolatedUnivariateSpline(betas, DEmin)
        spl = InterpolatedUnivariateSpline(betas, DE)
        splmax = InterpolatedUnivariateSpline(betas, DEmax)

        for betaid in range(nt):
            DeltaSmin[betaid,hid] =                splmin.integral(betas[0], betas[betaid])-                betas[betaid]*DEmin[betaid] + betas[0]*DEmin[0];
            DeltaS[betaid,hid] =            spl.integral(betas[0], betas[betaid])- betas[betaid]*DE[betaid]+ betas[0]*DE[0];
            DeltaSmax[betaid,hid] =            splmax.integral(betas[0], betas[betaid])- betas[betaid]*DEmax[betaid]+ betas[0]*DEmax[0];
      
            #Sshifted = S0- Int[bidmin:bidmax] + betas[bidmin:bidmax]*DE[bidmin:bidmax]- betas[bidmin]*DE[bidmin];
    
    t_h_Smin = S0*np.ones((nt, nh)) - DeltaSmax[::-1,:];
    t_h_S = S0*np.ones((nt, nh)) - DeltaS[::-1,:];
    t_h_Smax = S0*np.ones((nt, nh)) - DeltaSmin[::-1,:];


    return t_h_S, t_h_Smin, t_h_Smax


# In[ ]:


#def NewLoadEnergy(foldername, filenamelist, numsistes,
#                 nb, stat_temps, temperatures, stat_hfields,
#                 listfunction, **kwargs):
#    mergeruns = kwargs.get('mergeruns', False)
#    group = kwargs.get('groupruns', 1)
#    
#    assert group != 0
#    
#    t_h_MeanE = [[] for _ in range(n//group)]
#    t_h_MeanEsq = [[] for _ in range(n//group)]
#    t_h_varMeanE = [[] for _ in range(n//group)]
#    t_h_varMeanEsq = [[] for _ in range(n//group)]
#    
#    t_h_VarE= [[] for _ in range(n//group)]
#    t_h_ErrVarE= [[] for _ in range(n//group)]
#    C = [[] for _ in range(n//group)]
#    ErrC = [[] for _ in range(n//group)]
#    
#    for nrun in range(n//group):
#        nmin = group*nrun
#        nmax = group*(nrun+1)
#        
#        if 'Energy' in listfunctions[nmin]:
#            
#            idfunc = listfunctions[nmin].index('Energy')
#            [t_h_MeanE[nrun], t_h_MeanEsq[nrun], t_h_varMeanE[nrun], t_h_varMeanEsq[nrun],
#             t_h_VarE[nrun], t_h_ErrVarE[nrun], C[nrun], ErrC[nrun],
#             t_h_S[nrun], t_h_Smin[nrun], t_h_Smax[nrun]] =\
#            LoadEnergyFromRuns(nmin, nmax, foldername, filenamelist, numsites,
#                              nb, stat_temps, temperatures, stat_hfields, idfunc, **kwargs)


# In[ ]:


def LoadEnergyFromRuns(nmin, nmax, foldername, filenamelist, numsites,
                              nb, stat_temps, temperatures, stat_hfields, idfunc, **kwargs):
    
    ## Load, grouping by run
    backups = ["./"+foldername+filenamelist[nf] for nf in range(nmin, nmax)]
    
    mergeruns = kwargs.get('mergeruns', False)
    EGS = kwargs.get('EGS', 0)
    name = "Energy"
    
    nb_drop = kwargs.get('nb_drop', 0)
    jackknife = kwargs.get('jackknife', False)
    l = kwargs.get('binning_level', 0) # binning level how many bins together for specific heat
    nbb = nb//(2**l)
    nbb_drop = nb_drop//(2**l)
    t_h_MeanE, t_h_varMeanE =    ExtractStatisticsEnergy(nmin, nmax, backups, idfunc, name,
                      nb, stat_temps, stat_hfields, sq = 0, **kwargs)
    
    t_h_MeanEsq, t_h_varMeanEsq =    ExtractStatisticsEnergy(nmin, nmax, backups, idfunc, name, nb,
                      stat_temps, stat_hfields, sq = 1, **kwargs)

    
    name = "Energy"
    
    nb_drop = kwargs.get('nb_drop', 0)
    jackknife = kwargs.get('jackknife', False)
    l = kwargs.get('binning_level', 0) # binning level: how many bins together for specific heat
    
    r_bsth_E = [[] for _ in range(nmax-nmin)]
    nbtot = 0
    for r in range(nmax-nmin):
        r_bsth_E[r] = hkl.load(backups[r]+"_"+name+"_final.hkl")
        nbtot += nb[nmin+r]
    
    nt = len(stat_temps[nmin])
    nh = len(stat_hfields[nmin])
    print("LoadEnergyFromRuns: nh ", nh)
    
    bsth_E = np.array(r_bsth_E).reshape(nbtot, 2, nt, nh) # group all
    
    nbb = nbtot//(2**l)
    nbb_drop = nb_drop//(2**l)
    
    if not jackknife: # aim is to win a bit of time for quick overviews
        print("No jackknife analysis -- binning level: ", l, " -- number of bins: ", nbb-nbb_drop)
        C = np.zeros((nt, nh))
        VarE = (t_h_MeanEsq - t_h_MeanE**2)*((nbb - nbb_drop)/(nbb-nbb_drop-1)) # unbiased estimator
        print("LoadEnergyFromRuns: VarE.shape ", VarE.shape)
        if nt == len(temperatures[nmin]):
            print("LoadEnergyFromRuns: nt matches len(temperatures[nmin])")
            print("--numsites[nmin] ", numsites[nmin])
            print("--temperatures", (np.array(temperatures[nmin])[:,np.newaxis] ** 2).shape)
            C = numsites[nmin] * (VarE / (np.array(temperatures[nmin])[:,np.newaxis] ** 2))
        else:
            print("LoadEnergyFromRuns: nt DOESNT MATCH len(temperatures[nmin])")
            
            for resid, t in enumerate(stat_temps):
                C[resid,:] = numsites[nmin] * (VarE[t,:]/ (temperatures[nmin][t] ** 2))

        print("LoadEnergyFromRuns: C.shape ", C.shape)
        if np.any(VarE < 0) :
            err = np.max(np.abs(VarE - np.abs(VarE)))/2
            warnings.warn("Negative variances num error: %E" % err)
            # set to zero
            C[C<0] = 0
            VarE[VarE<0] = 0
        print(bsth_E.shape)
        
        # This below can still be made much more efficient but definitely not my priority right now
        ErrVarE = np.zeros((nt, nh))
        ErrC = np.zeros((nt, nh))
        if nt == len(temperatures[nmin]):
            Sum_VarE = np.zeros((nt, nh))
            Sum_VarE_Sq = np.zeros((nt, nh))
            
            for b in range(nbb_drop,nbb):
                Sum_VarE += (np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),1,:,:], axis = 0) - 
                                  np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),0,:,:], axis = 0) ** 2)
                Sum_VarE_Sq += ((np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),1,:,:], axis = 0) -
                                      np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),0,:,:], axis = 0) ** 2) ** 2)
            
            
            assert(np.all(Sum_VarE_Sq/(nbb-nbb_drop) - (Sum_VarE/(nbb-nbb_drop)) ** 2 >= -1e-15))
            ErrVarE = np.sqrt(((nbb-nbb_drop)/(nbb-nbb_drop-1)**3)*Sum_VarE_Sq - 1/(nbb-nbb_drop-1)**3 * Sum_VarE**2)
            # Momentaneously: we want the variance of the population, and not the error on the mean, because we expect it
            # could be shifted...
            #ErrVarE = np.sqrt(((nbb-nbb_drop)/(nbb-nbb_drop-1)**2)*Sum_VarE_Sq - 1/(nbb-nbb_drop-1)**2 * Sum_VarE**2)
         
            # So that everything lies within 2 sigmas:
            ErrVarE = 2*ErrVarE
            
            ErrC = ((numsites[nmin]) * ErrVarE)/(np.array(temperatures[nmin])[:,np.newaxis] ** 2)
            
        
        else:
            for resid, t in enumerate(stat_temps):
                T = temperatures[nmin][t]
                ErrCh = np.zeros(nh)
                ErrVarEh = np.zeros(nh)
                for reshid, h in enumerate(stat_hfields):

                    Sum_VarE = 0
                    Sum_VarE_Sq = 0
                    
                    
                    for b in range(nbb_drop,nbb):
                        Sum_VarE += (np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),1,resid,reshid], axis = 0) - 
                                          np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),0,resid,reshid], axis = 0) ** 2)
                        Sum_VarE_Sq += ((np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),1,resid,reshid], axis = 0) -
                                              np.mean(bsth_E[(2**l)*b:(2**l)*(b+1),0,resid,reshid], axis = 0) ** 2) ** 2)

                    if (Mean_VarE_Sq - Mean_VarE ** 2 >= 0) :
                        ErrVarEh[reshid] = np.sqrt(((nbb-nbb_drop)/(nbb-nbb_drop-1)**3)*Sum_VarE_Sq - 1/(nbb-nbb_drop-1)**3 * Sum_VarE**2)
                        ErrCh[reshid] = (numsites[nmin] / (T ** 2)) * ErrVarEh
                    else:
                        assert(Mean_VarE_Sq - Mean_VarE ** 2 >= -1e-15)

                ErrVarE[resid, :] = ErrVarEh
                ErrC[resid,:] = ErrCh
         
    else:
        print("Jackknife analysis -- binning level: ", l, " -- number of bins: ", nbb-nbb_drop)
        # Some time, write this as a standalone, for this is dependent of a function
        # of pre-computed variables
        # Evaluate the function on all but one segment (i.e. all bins but one), at the requested binning level
        # 1- evaluate on all bins
        VarE = np.zeros((nt, nh))
        ErrVarE = np.zeros((nt, nh))
        VarE0 = np.zeros((nt, nh))
        
        C = np.zeros((nt, nh))
        ErrC = np.zeros((nt, nh))
        C0 = np.zeros((nt, nh))
        
        # Estimator of the specific heat:
        if nt == len(temperatures[nmin]):
            VarE0 = (t_h_MeanEsq - t_h_MeanE**2)
            C0 = numsites[nmin] * (VarE0/ (np.array(temperatures[nmin])[:,np.newaxis] ** 2))
        else:
            for resid, t in enumerate(stat_temps):
                C0[t,:] = numsites[nmin] * (VarE0[t,:]/ (temperatures[nmin][t] ** 2))
        # 2 - for each bin, evaluate on all but this bin
         
        VarEs = np.zeros((nbb-nbb_drop, nt, nh))
        Cs = np.zeros((nbb-nbb_drop, nt, nh))
        #   load
        for b in range(nbb-nbb_drop):
            #print("b: ", b, " 2**l: ", 2**l)
            
            mE= np.mean(np.delete(bsth_E[:,0,:,:], np.arange((2**l)*(b+nbb_drop),(2**l)*(b+nbb_drop+1)), axis = 0), axis = 0) # mean over all but 1 bin
            mESq = np.mean(np.delete(bsth_E[:,1,:,:], np.arange((2**l)*(b+nbb_drop),(2**l)*(b+nbb_drop+1)), axis = 0), axis = 0)
            
            if nt == len(temperatures[nmin]):
                # shape: nbb, nt, nh
                VarEs[b,:,:] =  (mESq- mE **2)
                Cs[b,:,:] =  numsites[nmin] * (VarEs[b,:,:]/ (np.array(temperatures[nmin])[:, np.newaxis] ** 2))
            else:
                for resid, t in enumerate(stat_temps):
                    VarEs[b,resid,:]=(mESq[t,:]- mE[t,:] **2)
                    Cs[b,resid,:] = numsites[nmin] * (VarEs[b,resid,:]/ (temperatures[nmin][t] ** 2))
              
        # 3 - Compute C
        VarEdot = np.mean(VarEs,axis = 0)
        Cdot = np.mean(Cs, axis = 0);
        
        print("Cdot ", Cdot[0])
        print("C0", C0[0])
        
        VarE = VarE0 - (nbb - nbb_drop - 1)*(VarEdot - VarE0)
        C = C0 - (nbb - nbb_drop - 1)*(Cdot - C0)
        if not np.all(np.abs(C - C0)<1e-7):
            warnings.warn("Significant disagreement on C")
        
            plt.figure()
            plt.semilogx(temperatures[nmin], C0[:,0], label = 'C0')
            plt.semilogx(temperatures[nmin], Cdot[:,0], label = 'Cdot')
            plt.plot(temperatures[nmin], C[:,0], label = 'C')
            plt.legend()
            plt.show()


            plt.figure()
            plt.semilogx(temperatures[nmin], C0[:,0]-Cdot[:,0], label = 'C0 - Cdot')
            plt.semilogx(temperatures[nmin], C[:,0]-C0[:,0], label = 'C - C0')
            plt.legend()
            plt.show()
        
        
        # 4 - Compute the statistical error on C:
        print("Cs.shape : ", Cs.shape)
        print("Cdot : ", Cdot.shape)
        print("Cs-Cdot shape:", (Cs - Cdot).shape)
        
        VarEstimVarE = np.mean((VarEs-VarEdot)**2, axis = 0)
        ErrVarE = np.sqrt((nbb - nbb_drop - 1)*VarEstimVarE)
        VarEstimC = np.mean((Cs-Cdot)**2, axis = 0)
        ErrC = np.sqrt((nbb - nbb_drop - 1)*VarEstimC) # check this!
        print("Shape of ErrC : ", ErrC.shape)

         
    RS = kwargs.get('RS', False)
    mergeruns = kwargs.get('mergeruns', False)
    if RS: #else, we compute on the average C
        [t_h_S, t_h_Smin, t_h_Smax] =        ComputeEntropy(C, ErrC, temperatures[nmin], stat_hfields[nmin], nt, nh,**kwargs)
        [t_h_SE, t_h_SEmin, t_h_SEmax] =        ComputeEntropyFromE(t_h_MeanE, t_h_varMeanE, temperatures[nmin], stat_hfields[nmin], nt, nh,**kwargs)
    else:
        [t_h_S, t_h_Smin, t_h_Smax] = [[],[],[]]
    
    
    return t_h_MeanE, t_h_MeanEsq, t_h_varMeanE, t_h_varMeanEsq,    VarE, ErrVarE, C, ErrC, t_h_S, t_h_Smin, t_h_Smax,    t_h_SE, t_h_SEmin, t_h_SEmax


# In[ ]:


def ExtractStatisticsEnergy(nmin, nmax, backups, idfunc, name,
                             nb, stat_temps, stat_hfields, sq = 0, **kwargs):
    '''
        This function gets the statistics from a run saved over a few files and
        computes the expectation values and variances of 
        the operator corresponding to idfunc
        
        sq = 0 -> not square stats
        sq = 1 -> square stats
    '''
    nb_drop = kwargs.get('nb_drop', 0)
    
    nb_stattuples = [[] for r in range(nmax-nmin)]
    nbtot = 0
    for r in range(nmax-nmin):
        nb_stattuples[r] = hkl.load(backups[r]+"_"+name+"_final.hkl")
        nbtot += nb[nmin+r]
        
    l = kwargs.get('binning_level', 0) # binning level: how many bins together for specific heat
    
    nt = len(stat_temps[nmin])
    nh = len(stat_hfields[nmin])
    
    nb_stattuple = np.array(nb_stattuples).reshape(nbtot, 2, nt, nh) # group all
    
    t_h_meanfunc = nb_stattuple[nb_drop:nbtot].sum(0)/(nbtot-nb_drop)
    t_h_meanfunc = t_h_meanfunc[sq]
    
    
    binning = kwargs.get('binning', False)

    t_h_varmeanfunc = np.zeros(t_h_meanfunc.shape)
    #t_h_varmeanfunc_control = np.zeros(t_h_meanfunc.shape)
    
    t_h_varmeanfunc =  ((nb_stattuple[nb_drop:nbtot,sq,:,:] - t_h_meanfunc[np.newaxis,:,:])  ** 2).sum(0)/((nbtot -nb_drop)* (nbtot -nb_drop - 1))
        
    #for resid, t in enumerate(stat_temps):
    #    for reshid, h in enumerate(stat_hfields):
    #        for b in range(nb_drop, nb):
    #            t_h_varmeanfunc_control[resid,reshid] += ((nb_stattuple[b, sq, resid, reshid] - t_h_meanfunc[resid][reshid]) ** 2)
    #            # note that this is like t_h_varmeanfunc[resid,reshid, :]
    #        t_h_varmeanfunc_control[resid, reshid] = t_h_varmeanfunc_control[resid, reshid]/((nb -nb_drop)* (nb -nb_drop - 1))
        
    if binning:
        print("Binning..." + name)
        t_h_varmeanfunc = Binning(t_h_meanfunc,t_h_varmeanfunc, nb_stattuple[nb_drop:,sq,:,:], nbtot-nb_drop,
                                stat_temps[nmin], stat_hfields[nmin], name = name, sq = sq, **kwargs)
        
        if name == "FirstCorrelations":
            print(t_h_varmeanfunc[0][0])
    return t_h_meanfunc, t_h_varmeanfunc


# In[ ]:


def LoadEnergy(foldername, filenamelist, numsites,
               nb, stat_temps, temperatures, stat_hfields,
               listfunctions, **kwargs):
    
    n = len(filenamelist)
    
    
    mergeruns = kwargs.get('mergeruns', False)
    print("Merge runs?", mergeruns)
    group = kwargs.get('groupruns', 1)
    print("Group runs by...", group)
    assert group != 0
    
    t_h_MeanE = [[] for _ in range(n//group)]
    t_h_MeanEsq = [[] for _ in range(n//group)]
    t_h_varMeanE = [[] for _ in range(n//group)]
    t_h_varMeanEsq = [[] for _ in range(n//group)]
    
    t_h_VarE= [[] for _ in range(n//group)]
    t_h_ErrVarE= [[] for _ in range(n//group)]
    C = [[] for _ in range(n//group)]
    ErrC = [[] for _ in range(n//group)]
    
    t_h_S= [[] for _ in range(n//group)]
    t_h_Smin= [[] for _ in range(n//group)]
    t_h_Smax= [[] for _ in range(n//group)]
    
    t_h_SE= [[] for _ in range(n//group)]
    t_h_SEmin= [[] for _ in range(n//group)]
    t_h_SEmax= [[] for _ in range(n//group)]
    
    EGS = kwargs.get("EGSs",[0 for i in range(n//group)])
            
    for nrun in range(n//group):
        nmin = group*nrun
        nmax = group*(nrun+1)
        
        if 'Energy' in listfunctions[nmin]:
            
            idfunc = listfunctions[nmin].index('Energy')
            [t_h_MeanE[nrun], t_h_MeanEsq[nrun], t_h_varMeanE[nrun], t_h_varMeanEsq[nrun],
             t_h_VarE[nrun], t_h_ErrVarE[nrun], C[nrun], ErrC[nrun],
             t_h_S[nrun], t_h_Smin[nrun], t_h_Smax[nrun],
             t_h_SE[nrun], t_h_SEmin[nrun], t_h_SEmax[nrun]] =\
            LoadEnergyFromRuns(nmin, nmax, foldername, filenamelist, numsites,
                              nb, stat_temps, temperatures, stat_hfields, idfunc, EGS = EGS[nrun], **kwargs)
        else:
            [t_h_MeanE[nf], t_h_MeanEsq[nf], t_h_varMeanE[nf],
             t_h_varMeanEsq[nf], C[nf], ErrC[nf],
             t_h_S[nf], t_h_Smin[nf], t_h_Smax[nf]] = \
            [[],[],[],[],[],[],[],[],[]]
    
    print(t_h_MeanE[0].shape)
    print("Load Energy: C[0].shape; before merge", C[0].shape)
    ####
    [Merged_t_h_MeanE, Merged_t_h_MeanEsq, 
    Merged_t_h_varMeanE, Merged_t_h_varMeanEsq,
    Merged_t_h_VarE, Merged_t_h_ErrVarE,Merged_C, Merged_ErrC,
    Merged_t_h_S, Merged_t_h_Smin,Merged_t_h_Smax,
    Merged_t_h_SE, Merged_t_h_SEmin,Merged_t_h_SEmax] = \
    [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    newn = n//group
    if mergeruns:
        # Recompute averages
        for nf in range(newn):
            if nf == 0:
                Merged_t_h_MeanE = np.copy(t_h_MeanE[0])/newn
                Merged_t_h_MeanEsq = np.copy(t_h_MeanEsq[0])/newn
                Merged_C = np.copy(C[0])/newn
                Merged_t_h_VarE = np.copy(t_h_VarE[0])/newn
            else:
                Merged_t_h_MeanE += t_h_MeanE[nf]/newn
                Merged_t_h_MeanEsq += t_h_MeanEsq[nf]/newn
                Merged_C += C[nf]/newn
                Merged_t_h_VarE += t_h_VarE[nf]/newn
    
        # Recompute errors
        for nf in range(newn):
            if nf == 0:
                Merged_t_h_varMeanE = (t_h_MeanE[0]-Merged_t_h_MeanE)**2 / (newn*(newn-1))
                Merged_t_h_varMeanEsq = (t_h_MeanEsq[0]-Merged_t_h_MeanEsq)**2 / (newn*(newn-1))
                Merged_ErrC =(C[0]-Merged_C)**2 /(newn*(newn-1))
                Merged_t_h_ErrVarE = (t_h_VarE[0]-Merged_t_h_VarE)**2 /(newn*(newn-1))
            else:
                Merged_t_h_varMeanE += (t_h_MeanE[nf]-Merged_t_h_MeanE)**2 / (newn*(newn-1))
                Merged_t_h_varMeanEsq += (t_h_MeanEsq[nf]-Merged_t_h_MeanEsq)**2 / (newn*(newn-1))
                Merged_ErrC +=(C[nf]-Merged_C)**2 /(newn*(newn-1)) ## check if variance or std in normal case!!!
                Merged_t_h_ErrVarE +=(t_h_VarE[nf]-Merged_t_h_VarE)**2 /(newn*(newn-1)) ## check if variance or std in normal case!!!
                
        # So that ErrC is in the right form
        Merged_t_h_ErrVarE = 2*np.sqrt(Merged_t_h_ErrVarE)
        Merged_ErrC = 2*np.sqrt(Merged_ErrC)
        
        # Compute S:
        RS = kwargs.get('RS', False)
        nt = len(stat_temps[0])
        nh = len(stat_hfields[0])
        if RS: # if mergeruns, we compute on average C
            [Merged_t_h_S, Merged_t_h_Smin, Merged_t_h_Smax] =            ComputeEntropy(Merged_C, Merged_ErrC, temperatures[0], stat_hfields[0], nt, nh,**kwargs)
            [Merged_t_h_SE, Merged_t_h_SEmin, Merged_t_h_SEmax] =            ComputeEntropyFromE(Merged_t_h_MeanE, Merged_t_h_varMeanE,
                                temperatures[0], stat_hfields[0], nt, nh,
                                EGS = EGS[0], **kwargs)
    
    MergedData = [[Merged_t_h_MeanE], [Merged_t_h_MeanEsq], 
        [Merged_t_h_varMeanE], [Merged_t_h_varMeanEsq],[Merged_t_h_VarE],[Merged_t_h_ErrVarE],
        [Merged_C],[Merged_ErrC],
        [Merged_t_h_S],[Merged_t_h_Smin],[Merged_t_h_Smax],
        [Merged_t_h_SE],[Merged_t_h_SEmin],[Merged_t_h_SEmax]]
    
    #####
    
    return t_h_MeanE, t_h_MeanEsq, t_h_varMeanE, t_h_varMeanEsq,  t_h_VarE, t_h_ErrVarE,C, ErrC,            t_h_S, t_h_Smin, t_h_Smax, t_h_SE, t_h_SEmin, t_h_SEmax, MergedData


# In[ ]:


def LoadMagnetisation(foldername, filenamelist, numsites, nb, stat_temps,
                      temperatures, stat_hfields, listfunctions, **kwargs):
    n = len(filenamelist)
    
    t_h_MeanM = [[] for _ in range(n)]
    t_h_MeanMsq = [[] for _ in range(n)]
    t_h_varMeanM = [[] for _ in range(n)]
    t_h_varMeanMsq = [[] for _ in range(n)]
    Chi = [[] for _ in range(n)]
    ErrChi = [[] for _ in range(n)]

    for nf, filename in enumerate(filenamelist):
        if 'Magnetisation' in listfunctions[nf]:
            idfunc = listfunctions[nf].index('Magnetisation')
            [t_h_MeanM[nf], t_h_MeanMsq[nf], t_h_varMeanM[nf],
             t_h_varMeanMsq[nf], Chi[nf], ErrChi[nf]] = \
                LoadMagnetisationFromFile(foldername, filename, numsites[nf],
                                          nb[nf], stat_temps[nf], temperatures[nf],
                                          stat_hfields[nf], idfunc, **kwargs)
        else:
            [t_h_MeanM[nf], t_h_MeanMsq[nf], t_h_varMeanM[nf],
             t_h_varMeanMsq[nf], Chi[nf], ErrChi[nf]] = [[],[],[],[],[],[]]
        
    return t_h_MeanM, t_h_MeanMsq, t_h_varMeanM, t_h_varMeanMsq, Chi, ErrChi


# In[ ]:


def LoadMagnetisationFromFile(foldername, filename, numsites, nb, stat_temps,
                              temperatures, stat_hfields, idfunc,  **kwargs):
    
    backup = "./"+foldername+filename
    name = "Magnetisation"
    
    t_h_MeanM, t_h_varMeanM =    ExtractStatistics(backup, idfunc, name,nb, stat_temps,
                      stat_hfields, **kwargs)
    t_h_MeanMsq, t_h_varMeanMsq =    ExtractStatistics(backup, idfunc, name,nb, stat_temps,
                      stat_hfields, sq = 1, **kwargs)
    Chi = []
    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        Chih = []
        for reshid, h in enumerate(stat_hfields):
            Chih.append(numsites * ( t_h_MeanMsq[resid][reshid] -  t_h_MeanM[resid][reshid] ** 2) / T)
        Chi.append(Chih)
        
    bsth_M = hkl.load(backup+"_"+name+"_final.hkl")

    
    ErrChi = []
    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        ErrChih = []
        for reshid, h in enumerate(stat_hfields):
            Mean_VarM = 0
            Mean_VarM_Sq = 0
            for b in range(nb):
                    Mean_VarM += (bsth_M[b][1][resid][reshid]
                                  - bsth_M[b][0][resid][reshid] ** 2)/nb
                    Mean_VarM_Sq += ((bsth_M[b][1][resid][reshid]
                                      - bsth_M[b][0][resid][reshid] ** 2) ** 2)/nb
            if Mean_VarM_Sq - Mean_VarM ** 2 >= 0 :
                ErrChih.append((numsites / T) * np.sqrt((Mean_VarM_Sq - Mean_VarM ** 2)/(nb-1)))  
            else:
                assert(Mean_VarM_Sq - Mean_VarM ** 2 >= -1e-15)
                ErrChih.append(0)
        ErrChi.append(ErrChih)
    
    return  t_h_MeanM,  t_h_MeanMsq, t_h_varMeanM, t_h_varMeanMsq, Chi, ErrChi


# In[ ]:


def LoadFirstCorrelations(foldername, filenamelist, listfunctions, stat_temps,
                          stat_hfields, nb,t_h_MeanM, t_h_varMeanMsq, **kwargs):
    n = len(filenamelist)
    
    ## "First Correlations" (check!!)
    t_h_MeanFc = [[] for _ in range(n)]
    t_h_varMeanFc = [[] for _ in range(n)]
    
    ## Local spin average
    #t_h_MeanSi = [[] for _ in range(n)]
    #t_h_varMeanSi = [[] for _ in range(n)]
     
    for nf, filename in enumerate(filenamelist):
        if ('FirstCorrelations' in listfunctions[nf]):
            # This will be improved when we will work with a more
            # general way of handling the correlations
            idfunc = listfunctions[nf].index('FirstCorrelations')
            #idfuncsi = listfunctions[nf].index('Si')

            [t_h_MeanFc[nf], t_h_varMeanFc[nf]] =             LoadFirstCorrelationsFromFile(foldername, filename, idfunc,
                                    stat_temps[nf],
                                     stat_hfields[nf], nb[nf],t_h_MeanM[nf], t_h_varMeanMsq[nf], **kwargs)
        else:
            [t_h_MeanFc[nf], t_h_varMeanFc[nf]] = [[],[]]
    return t_h_MeanFc, t_h_varMeanFc


# In[ ]:


def LoadFirstCorrelationsFromFile(foldername, filename, idfunc, 
                                  stat_temps, 
                                  stat_hfields, nb, t_h_MeanM,
                                  t_h_varMeanMsq,**kwargs):
    
    backup = "./"+foldername+filename
    name = "FirstCorrelations"
    #namesi = "Si"
    rmmag = kwargs.get('rmmag', False)
    
    # Averages and corresponding variances
    #t_h_MeanSi, t_h_varMeanSi =\
    #ExtractStatistics(backup, idfuncsi, namesi, nb, stat_temps,
    #                  stat_hfields, **kwargs)
    
    t_h_MeanFc, t_h_varMeanFc =    ExtractStatistics(backup, idfunc, name, nb, stat_temps,
                      stat_hfields, sq=0, **kwargs)
    
    print(t_h_MeanFc.shape)
    #print(t_h_MeanSi.shape)
    if rmmag:
        m = t_h_MeanM
        #m = t_h_MeanSi.sum(2)/t_h_MeanSi.shape[2] # sample average
        for nni in range(t_h_MeanFc.shape[2]):
            t_h_MeanFc[:,:,nni] = (t_h_MeanFc[:,:,nni] - m**2) #<si sj> - <si> <sj> for j in lattice. /!\ this is ELEMENTWISE
            
    for i in range(t_h_varMeanFc.shape[2]):
        t_h_varMeanFc[:,:,i] = t_h_varMeanFc[:,:,i]+ t_h_varMeanMsq[:,:] # approximately
    
    return t_h_MeanFc, t_h_varMeanFc#, t_h_MeanSi, t_h_varMeanSi


# In[ ]:


def LoadFrustratedTriangles(foldername, filenamelist, listfunctions,
                stat_temps, stat_hfields, nb, **kwargs):
    
    n = len(filenamelist)
    ## Charges
    t_h_MeanCharges = [[] for _ in range(n)]
    t_h_varMeanCharges = [[] for _ in range(n)]
    
    for nf, filename in enumerate(filenamelist):
        if ('FrustratedTriangles' in listfunctions[nf]):
            idfunc = listfunctions[nf].index('FrustratedTriangles')
            
            [t_h_MeanCharges[nf], t_h_varMeanCharges[nf]] =             LoadFrustratedTrianglesFromFile(foldername, filename, idfunc,
                               stat_temps[nf], stat_hfields[nf],
                                nb[nf], **kwargs)
            
        else:
            [t_h_MeanCharges[nf], t_h_varMeanCharges[nf]] = [[],[]]
            
        
    return t_h_MeanCharges, t_h_varMeanCharges


# In[ ]:


def LoadFrustratedTrianglesFromFile(foldername, filename, idfunc, 
                        stat_temps, stat_hfields, nb, **kwargs):
    
    backup = "./"+foldername+filename
    name = "FrustratedTriangles"
    
    t_h_MeanCharges, t_h_varMeanCharges =    ExtractStatistics(backup, idfunc, name, nb, stat_temps,
                      stat_hfields, **kwargs)
    
    
    return t_h_MeanCharges, t_h_varMeanCharges


# In[ ]:


def LoadCentralCorrelations(foldername, filenamelist, listfunctions, srefs, stat_temps, stat_hfields, nb, t_h_MeanM, **kwargs):
    n = len(filenamelist)
    
    ## "Correlations" <sisj>
    t_h_MeanSs = [[] for _ in range(n)]
    t_h_varMeanSs = [[] for _ in range(n)]
    
    ### Local spin average
    #t_h_MeanSi = [[] for _ in range(n)]
    #t_h_varMeanSi = [[] for _ in range(n)]
    
    ## Correlations
    t_h_MeanCorr = [[] for _ in range(n)]
    t_h_errCorrEstim = [[] for _ in range(n)]
    for nf, filename in enumerate(filenamelist):
        if ('Central_Correlations' in listfunctions[nf]):
            # This will be improved when we will work with a more
            # general way of handling the correlations
            idfunc = listfunctions[nf].index('Central_Correlations')
            #idfuncsi = listfunctions[nf].index('Si')

            [t_h_MeanSs[nf], t_h_varMeanSs[nf], t_h_MeanCorr[nf],t_h_errCorrEstim[nf]] =            LoadCorrelationsFromFile(foldername, filename, idfunc,
                                     srefs[nf], stat_temps[nf],
                                     stat_hfields[nf], nb[nf],t_h_MeanM[nf], **kwargs)
        else:
            [t_h_MeanSs[nf], t_h_varMeanSs[nf], t_h_MeanCorr[nf], 
             t_h_errCorrEstim[nf]] = [[],[],[],[]]
    return t_h_MeanSs, t_h_varMeanSs,t_h_MeanCorr, t_h_errCorrEstim # t_h_MeanSi, t_h_varMeanSi, 


# In[ ]:


def LoadCorrelationsFromFile(foldername, filename, idfunc, srefs, 
                             stat_temps, stat_hfields, nb, t_h_MeanM,**kwargs):
    
    backup = "./"+foldername+filename
    name = "Central_Correlations"
    #namesi = "Si"
    rmmag = kwargs.get('rmmag', False)
    
    # Averages and corresponding variances
    #t_h_MeanSi, t_h_varMeanSi =\
    #ExtractStatistics(backup, idfuncsi, namesi, nb, stat_temps,
    #                  stat_hfields, **kwargs)
    
    t_h_MeanSs, t_h_varMeanSs =    ExtractStatistics(backup, idfunc, name, nb, stat_temps,
                      stat_hfields, **kwargs)
    
    t_h_MeanCorr = []
    assert len(srefs) == 3
    for i in range(len(srefs)):
        
        if rmmag:
            t_h_MeanCorr.append(t_h_MeanSs[:,:,i,:] - t_h_MeanM[:,:,np.newaxis]**2) #<si sj> - <si> <sj> for j in lattice. /!\ this is ELEMENTWISE
        else:
            t_h_MeanCorr.append(t_h_MeanSs[:,:,i,:])
            
    # Estimating the error on <si sj> - <si><sj>
    t_h_errCorrEstim = CorrelErrorEstimator(backup, idfunc,
                                            srefs,
                                            name, t_h_MeanM,
                                            nb)   

    return t_h_MeanSs, t_h_varMeanSs, np.array(t_h_MeanCorr), np.array(t_h_errCorrEstim)


# In[ ]:


def CorrelErrorEstimator(backup, idfunc, sref,
                         name, t_h_MeanM, nb):
    
    bsth_sisj = hkl.load(backup+"_"+name+"_final.hkl")
    bth_sisj = bsth_sisj[:,0,:,:,:,:] # <s0sj>_b (t,h)
    (nb, ntm, nhm, nrefs, nsites) = bth_sisj.shape

    sbth_gamma = np.zeros((nrefs,nb,ntm, nhm, nsites))
    sth_gamma = np.zeros((nrefs,ntm, nhm, nsites))
    for i in range(nrefs):
        
        sbth_gamma[i] = bth_sisj[:,:,:,i,:] - t_h_MeanM[:,:,np.newaxis]**2
        sth_gamma[i] = sbth_gamma[i].sum(0)/nb
    
    
    t_h_vargamma = np.zeros((nrefs,ntm, nhm, nsites))
    for i in range(nrefs):
        for b in range(nb):
            t_h_vargamma[i] += np.power((sbth_gamma[i][b] - sth_gamma[i]),2)
        t_h_vargamma[i] = t_h_vargamma[i]/(nb*(nb-1))

    return t_h_vargamma


# In[ ]:


#def LoadSi(foldername, filenamelist, listfunctions, **kwargs):
#    n = len(filenamelist)
#    
#    t_MeanSi = [[] for _ in range(n)]
#    t_varMeanSi = [[] for _ in range(n)]
#
#    for nf, filename in enumerate(filenamelist):
#        if 'Si' in listfunctions[nf]:
#            idfunc = listfunctions[nf].index('Si')
#            [t_MeanSi[nf], t_varMeanSi[nf]] = LoadSiFromFile(foldername, filename, idfunc,stat_temps[nf], **kwargs)
#        else:
#            [t_MeanSi[nf], t_varMeanSi[nf]] = [[],[]]
#            
#    return t_MeanSi, t_varMeanSi


# In[ ]:


#def LoadSiFromFile(foldername, filename, idfunc, stat_temps, **kwargs):
#    f = open('./' + foldername + filename +'.pkl', 'rb')
#    backup = pickle.load(f) 
#    
#    statstable = backup.results.statstable
#    
#    t_MeanSi, t_varMeanSi = ExtractStatistics(backup, idfuncsi, name, nb, stat_temps, **kwargs)
#    
#    f.close()
#    
#    return t_MeanSi, t_varMeanSi
#    


# In[ ]:


def SwapsAnalysis(L, n, tidmin, tidmax, temperatures, hfields, foldername, results_foldername, swapst, swapsh):

    for i in range(n):
        plt.figure()
        plt.loglog(temperatures[i][tidmin:tidmax[i]-1], swapst[i][tidmin:tidmax[i]-1], '.', color = 'green')
        plt.xlabel('Temperature')
        plt.ylabel('Ratio of swaps')
        plt.title('Ratio of swaps as a function of the temperature')
        plt.savefig('./' + foldername  + results_foldername+ '/NumberSwapsTemperature_L={0}_SimId={1}.png'.format(L[i],i))

        nh = len(hfields[i])
        if nh > 1:
            plt.figure()
            plt.semilogy(hfields[i], swapsh[i], '.', color = 'orange')
            plt.xlabel('Magnetic field')
            plt.ylabel('Ratio of swaps')
            plt.title('Ratio of swaps as a function of the magnetic field')
            plt.grid(which='both')
            plt.savefig('./' + foldername  + results_foldername+ '/NumberSwapsField_L={0}_SimId={1}.png'.format(L[i], i))


# In[ ]:


def FailedAnalysis(L, n, tidmin, tidmax, temperatures, hfields, foldername, results_foldername, failed, failedssf):

    for i in range(n):
        plt.figure()
        plt.semilogx(temperatures[i][tidmin:tidmax[i]-1], failed[i][tidmin:tidmax[i]-1], '.',label = 'worms')
        plt.semilogx(temperatures[i][tidmin:tidmax[i]-1], failedssf[i][tidmin:tidmax[i]-1], '.', label = 'ssf')
        plt.xlabel('Temperature')
        plt.ylabel('Ratio of failed attemps')
        plt.legend()
        plt.title('Ratio of failed attempts as a function of the temperature')
        plt.savefig('./' + foldername  + results_foldername+ '/NumberFailedAttempsTemperature_L={0}_SimId={1}.png'.format(L[i],i))
    


# In[ ]:


def testPhase(energy, modelenergy):
    if abs(energy - modelenergy) < 1e-6:
        return True
    elif energy < modelenergy:
        return False
    elif energy > modelenergy:
        return "Problem!!"


# In[ ]:


def BasicPlotsFirstCorrelations(L, i, t_h_MeanFc, temperatures_plots, t_h_varMeanFc,
                                foldername, results_foldername, filenamelist, 
                                tmin = 0, setyticks = None, addtitle = "", addsave = "",
                                save = True, log = True,
                                figsize=(11,9), dpi = 200, ax = [],hid = 0, **kwargs):
    
    createfig = kwargs.get('createfig', True)
    markersize = kwargs.get('markersize', 10)
    alpha = kwargs.get('alpha', 0.3)
    TNvals = kwargs.get('TNvals', [])
    
    print(createfig)
    if createfig:
        fig, ax = plt.subplots(1,1,figsize=figsize, dpi = dpi)
    ## NN
    if log:
        ax.semilogx(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,0],'.',markersize = markersize,label = r'$c_1$')
    else:
        ax.plot(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,0],'.',markersize = markersize,label = r'$c_1$')
    
    ax.fill_between(temperatures_plots[i][tmin:],
                    t_h_MeanFc[i][tmin:,hid,0]-np.sqrt(t_h_varMeanFc[i][tmin:,hid,0]),
                    t_h_MeanFc[i][tmin:,hid,0]+np.sqrt(t_h_varMeanFc[i][tmin:,hid,0]), alpha = alpha)
   
    if len(TNvals)>=2:
        c = TNvals[1];
        ax.plot(temperatures_plots[i][tmin], c,'o', color = "blue", markersize = markersize,
                label = r"TN value $c_{1}$")
    ## N2
    if log:
        ax.semilogx(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,1],'x',markersize = markersize,label = r'$c_2$')
    else: 
        ax.plot(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,1],'x',markersize = markersize,label = r'$c_2$')
    
    ax.fill_between(temperatures_plots[i][tmin:],
                    t_h_MeanFc[i][tmin:,hid,1]-np.sqrt(t_h_varMeanFc[i][tmin:,hid,1]),
                    t_h_MeanFc[i][tmin:,hid,1]+np.sqrt(t_h_varMeanFc[i][tmin:,hid,1]), alpha = alpha)
    if len(TNvals)>=3:
        c = TNvals[2];
        ax.plot(temperatures_plots[i][tmin], c,'o', color = "orange", markersize = markersize,
                label = r"TN value $c_{2}$") 
    ## N3||
    if log:
        ax.semilogx(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,2],'v',markersize = markersize,label = r'$c_{3||}$')
    else:
        ax.plot(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,2],'v',markersize = markersize,label = r'$c_{3||}$')
    ax.fill_between(temperatures_plots[i][tmin:],
                    t_h_MeanFc[i][tmin:,hid,2]-np.sqrt(t_h_varMeanFc[i][tmin:,hid,2]),
                    t_h_MeanFc[i][tmin:,hid,2]+np.sqrt(t_h_varMeanFc[i][tmin:,hid,2]), alpha = alpha)
    
    if len(TNvals)>=4:
        c = TNvals[3];
        ax.plot(temperatures_plots[i][tmin], c,'o', color = "green", markersize = markersize,
                label = r"TN value $c_{3||}$") 
    ## N3*
    if log:
        ax.semilogx(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,3],'*',markersize = markersize,label =r'$c_{3\star}$ ')
    else:
        ax.plot(temperatures_plots[i][tmin:],t_h_MeanFc[i][tmin:,hid,3],'*',markersize = markersize,label =r'$c_{3\star}$ ')
    
    ax.fill_between(temperatures_plots[i][tmin:],
                    t_h_MeanFc[i][tmin:,hid,3]-np.sqrt(t_h_varMeanFc[i][tmin:,hid,3]),
                    t_h_MeanFc[i][tmin:,hid,3]+np.sqrt(t_h_varMeanFc[i][tmin:,hid,3]), alpha = alpha)
   
    if len(TNvals)>=5:
        c = TNvals[4];
        ax.plot(temperatures_plots[i][tmin], c,'o', color = "red", markersize = markersize,
                label = r"TN value $c_{3\star}$") 
    if createfig:
        ax.set_title(addtitle)
        ax.set_xlabel(r"$T/J_1$")
        ax.set_ylabel(r"$\langle \sigma_i \sigma_j \rangle - \langle \sigma_i \rangle \langle \sigma_j \rangle$")
        ax.set_yticks(setyticks)
        ax.grid(which='both')
        ax.legend()
    if save:
        if log:
            plt.savefig("./" + foldername + results_foldername + "/FirstCorrelations"+addsave+ ".png")
        else:
            plt.savefig("./" + foldername + results_foldername + "/FirstCorrelations"+addsave+ "_Linear.png")


# In[ ]:


def BasicPlotsDifferenceFirstCorrelations(L, i, t_h_MeanFc, temperatures_plots, t_h_varMeanFc,
                                foldername, results_foldername, filenamelist, 
                                tmin = 0, tmax = 128, setxlim = None, setylim = None, 
                                setxticks = None, setyticks = None, 
                                addtitle = "", addsave = "",
                                save = True, log = True, 
                                figsize=(11,9), dpi = 200,**kwargs):
    plt.figure(figsize=figsize, dpi = dpi)
    if log:
        plt.semilogx(temperatures_plots[i][tmin:tmax],t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,2],'k.',label = r'$c_2$ - $c_{3||}$')
    else:
        plt.plot(temperatures_plots[i][tmin:tmax],t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,2],'k.',label = r'$c_2$ - $c_{3||}$')
    plt.fill_between(temperatures_plots[i][tmin:tmax],
                    t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,2]-2*np.sqrt(t_h_varMeanFc[i][tmin:,0,2]),
                    t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,2]+2*np.sqrt(t_h_varMeanFc[i][tmin:,0,2]), color = 'k', alpha = 0.2)
    if log:
        plt.semilogx(temperatures_plots[i][tmin:tmax],t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,3],'r.',label = r'$c_2$ - $c_{3\star}$ ')
    else:
        plt.plot(temperatures_plots[i][tmin:tmax],t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,3],'r.',label = r'$c_2$ - $c_{3\star}$ ')
    plt.fill_between(temperatures_plots[i][tmin:tmax],
                    t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,3]-2*np.sqrt(t_h_varMeanFc[i][tmin:,0,3]),
                    t_h_MeanFc[i][tmin:tmax,0,1]-t_h_MeanFc[i][tmin:tmax,0,3]+2*np.sqrt(t_h_varMeanFc[i][tmin:,0,3]), color = 'r', alpha = 0.2)

    plt.title(addtitle)
    plt.xlabel(r"$T/J_1$")
    plt.ylabel(r"$\Delta(\langle \sigma_i \sigma_j \rangle - \langle \sigma_i \rangle \langle \sigma_j \rangle)$")
    plt.xlim(setxlim)
    plt.ylim(setylim)
    plt.xticks(setxticks)
    plt.yticks(setyticks)
    plt.grid(which='both')
    plt.legend()
    if save:
        plt.savefig("./" + foldername + results_foldername + "/FirstCorrelationsDifference"+ addsave + "_ZoomLinear.png")


# In[ ]:


def BasicPlotsTriangles(L, n, tidmin, tidmax, temperatures_plots, hfields_plots, foldername,
                results_foldername, filenamelist, t_h_MeanFrustratedTriangles,
                        t_h_varMeanFrustratedTriangles, **kwargs):
    ploth = kwargs.get('ploth', False)
    pgf = kwargs.get('pgf', False)
    
    t_h_MeanFrustratedTriangles = np.array(t_h_MeanFrustratedTriangles)
    t_h_varMeanFrustratedTriangles =  np.array(t_h_varMeanFrustratedTriangles)
    
    margin = [0.08, 0.08, 0.02, 0.1]
    for i in range(n):
        if ploth:
            mt = tidmax[i];
            plt.figure(figsize=(12, 8),dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for tid, t in enumerate(temperatures_plots[i]):
                if tid >= tidmin and tid <= tidmax[i]:
                    col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                    plt.plot(hfields_plots[i],
                                     t_h_MeanFrustratedTriangles[i][tid, :],'.',\
                                      label = r'$T$ = {0}'.format(t), color = col)
                    plt.fill_between(hfields_plots[i],
                                     (t_h_MeanFrustratedTriangles[i][tid,:]
                                      - np.sqrt(t_h_varMeanFrustratedTriangles[i][tid,:])),
                                     (t_h_MeanFrustratedTriangles[i][tid,:]
                                      + np.sqrt(t_h_varMeanFrustratedTriangles[i][tid,:])),\
                                     alpha=0.4, color = col)
            plt.xlabel(r'Magnetic field $h$')
            plt.ylabel(r'$n_{fr.}/n_{t}$')
            plt.grid(which='both')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.savefig('./' + foldername  + results_foldername
                        + '/h_nfr.png')
            if pgf:
                plt.savefig('./' + foldername  + results_foldername
                        + '/h_nfr.pgf')
        else:
            mh = len(hfields_plots[i])
            plt.figure(figsize=(12, 8),dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                                 t_h_MeanFrustratedTriangles[i][tidmin:tidmax[i]][:,hid],'.',\
                                  label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                                 (t_h_MeanFrustratedTriangles[i][tidmin:tidmax[i]][:,hid]
                                  - np.sqrt(t_h_varMeanFrustratedTriangles[i][tidmin:tidmax[i]][:,hid])),
                                 (t_h_MeanFrustratedTriangles[i][tidmin:tidmax[i]][:,hid]
                                  + np.sqrt(t_h_varMeanFrustratedTriangles[i][tidmin:tidmax[i]][:,hid])),\
                                 alpha=0.4, color = col)
            plt.xlabel(r'Temperature $T$')
            plt.ylabel(r'$n_{fr.}/n_{t}$')
            plt.grid(which='both')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.savefig('./' + foldername  + results_foldername +                        '/t_nfr.png')
            if pgf:
                plt.savefig('./' + foldername  + results_foldername +                        '/t_nfr.pgf')


# In[ ]:


def BasicPlotsE(L, i, tidmin, tidmax, temperatures_plots, hfields_plots, foldername,
                results_foldername, filenamelist, t_h_MeanE, t_h_MeanEsq, t_h_varMeanE,
                t_h_varMeanEsq, C, ErrC, J1, J2, J3, J4, S0 = np.log(2), **kwargs):
    
    ploth = kwargs.get('ploth', False)
    pgf = kwargs.get('pgf', False)
    addsave = kwargs.get('addsave', "")
    alpha = kwargs.get('alpha', 0.2)
    
    t_h_MeanE = np.array(t_h_MeanE)
    t_h_MeanEsq =  np.array(t_h_MeanEsq)
    t_h_varMeanE =  np.array(t_h_varMeanE)
    t_h_varMeanEsq =  np.array(t_h_varMeanEsq)
    C = np.array(C)
    ErrC = np.array(ErrC)
    RS = kwargs.get('RS', False)
    if RS: 
        (t_h_S, t_h_Smin, t_h_Smax) = kwargs.get("t_h_EntropyData", [[],[],[]])   
    # Mean E
    margin = [0.08, 0.08, 0.02, 0.1]
    if ploth:
        mt = tidmax[i];
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for tid, t in enumerate(temperatures_plots[i]):
            if tid >= tidmin and tid <= tidmax[i]:
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i],
                                 t_h_MeanE[i][tid, :],'.',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i],
                                 (t_h_MeanE[i][tid,:]
                                  - np.sqrt(t_h_varMeanE[i][tid,:])),
                                 (t_h_MeanE[i][tid,:]
                                  + np.sqrt(t_h_varMeanE[i][tid,:])),\
                                 alpha=0.4, color = col)
        plt.xlabel(r'Magnetic field $h$')
        plt.ylabel(r'$E$')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername
                    + '/h_E_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername
                    + '/h_E_simId={0}.pgf'.format(i))
    else:
        mh = len(hfields_plots[i])
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for hid, h in enumerate(hfields_plots[i]):
            col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                             t_h_MeanE[i][tidmin:tidmax[i]][:,hid],'.',\
                              label = r'$h$ = {0}'.format(h), color = col)
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             (t_h_MeanE[i][tidmin:tidmax[i]][:,hid]
                              - np.sqrt(t_h_varMeanE[i][tidmin:tidmax[i]][:,hid])),
                             (t_h_MeanE[i][tidmin:tidmax[i]][:,hid]
                              + np.sqrt(t_h_varMeanE[i][tidmin:tidmax[i]][:,hid])),\
                             alpha=0.4, color = col)
        plt.xlabel(r'Temperature $T$')
        plt.ylabel(r'$E$')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername +                    '/Mean energy per site_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername +                    '/Mean energy per site_simId={0}.pgf'.format(i))

    
    #Heat capacity
    margin = [0.08, 0.08, 0.02, 0.1]

    if ploth:
        mt = tidmax[i]
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for tid, t in enumerate(temperatures_plots[i]):
            if tid >= tidmin and tid <= tidmax[i]:
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i],
                         C[i][tid,:],'.',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i],
                                 ( C[i][tid,:]
                                  - ErrC[i][tid,:]),
                                 ( C[i][tid,:]
                                  + ErrC[i][tid,:]),\
                                 alpha=0.4, color = col)
        plt.xlabel(r'Magnetic field $h$')
        plt.ylabel(r'Heat capacity $C$ ')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityErrors_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityErrors_simId={0}.pgf'.format(i))
                
        hidmin = kwargs.get('hidmin',0)
        hidmax = kwargs.get('hidmax',len(hfields_plots[0]) )
        
        mt = tidmax[i]
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for tid, t in enumerate(temperatures_plots[i]):
            if tid >= tidmin and tid <= tidmax[i]:
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i][hidmin:hidmax],
                         C[i][tid,hidmin:hidmax],'.',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i][hidmin:hidmax],
                                 ( C[i][tid,hidmin:hidmax]
                                  - ErrC[i][tid,hidmin:hidmax]),
                                 ( C[i][tid,hidmin:hidmax]
                                  + ErrC[i][tid,hidmin:hidmax]),\
                                 alpha=0.4, color = col)
        plt.xlabel(r'Magnetic field $h$')
        plt.ylabel(r'Heat capacity $C$ ')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityErrorsZoom_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityErrorsZoom_simId={0}.pgf'.format(i))

        
        mt = tidmax[i]
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for tid, t in enumerate(temperatures_plots[i]):
            print(t)
            if tid >= tidmin and tid <= tidmax[i]:
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i],
                         C[i][tid,:]/t,'.',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i],
                                 ( C[i][tid,:]/t
                                  - ErrC[i][tid,:]/t),
                                 ( C[i][tid,:]/t
                                  + ErrC[i][tid,:]/t),\
                             alpha=0.4, color = col)
        plt.xlabel(r'Magnetic field $h$')
        plt.ylabel(r'Heat capacity $C/T$ ')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityOverTErrors_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityOverTErrors_simId={0}.pgf'.format(i))
        
        mt = tidmax[i]
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for tid, t in enumerate(temperatures_plots[i]):
            if tid >= tidmin and tid <= tidmax[i]:
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i][hidmin:hidmax],
                         C[i][tid,hidmin:hidmax]/t,'.',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i][hidmin:hidmax],
                                 ( C[i][tid,hidmin:hidmax]/t
                                  - ErrC[i][tid,hidmin:hidmax]/t),
                                 ( C[i][tid,hidmin:hidmax]/t
                                  + ErrC[i][tid,hidmin:hidmax]/t),\
                                 alpha=0.4, color = col)
        plt.xlabel(r'Magnetic field $h$')
        plt.ylabel(r'Heat capacity $C$ ')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityErrorsOvTZoom_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/h_HeatCapacityErrorsOvTZoom_simId={0}.pgf'.format(i))

    else:
        mh = len(hfields_plots[i])
        plt.figure(figsize=(12, 8), dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for hid, h in enumerate(hfields_plots[i]):
            col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                         C[i][tidmin:tidmax[i]][:,hid], '.',\
                         label = r'$h$ = {0}'.format(h), color = col)
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             C[i][tidmin:tidmax[i]][:,hid]
                             - ErrC[i][tidmin:tidmax[i]][:,hid],
                             C[i][tidmin:tidmax[i]][:,hid]
                             + ErrC[i][tidmin:tidmax[i]][:,hid],\
                             alpha = alpha, color = col)
            #print('Error on the heat capacity for file ', filenamelist[i])
            #print(ErrC[i])
        plt.xlabel(r'Temperature $T$ ')
        plt.ylabel(r'Heat capacity $C$ ')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername+ '/HeatCapacityErrors_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/HeatCapacityErrors_simId={0}.pgf'.format(i))

        ##Heat capacity / T
        margin = [0.08, 0.08, 0.02, 0.1]

        mh = len(hfields_plots[i])
        plt.figure(figsize=(12, 8), dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for hid, h in enumerate(hfields_plots[i]):
            col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                         C[i][tidmin:tidmax[i]][:,hid] / temperatures_plots[i][tidmin:tidmax[i]],
                         '.', label = r'$h$ = {0}'.format(h), color = col)
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             (C[i][tidmin:tidmax[i]][:,hid]
                              - ErrC[i][tidmin:tidmax[i]][:,hid]
                             )/temperatures_plots[i][tidmin:tidmax[i]],
                             (C[i][tidmin:tidmax[i]][:,hid]
                              + ErrC[i][tidmin:tidmax[i]][:,hid]
                             )/temperatures_plots[i][tidmin:tidmax[i]],\
                             alpha = alpha, color = col)
        plt.xlabel(r'Temperature $T$ ')
        plt.ylabel(r'$\frac{c}{k_B T}$')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha=0.5)
        plt.savefig('./' + foldername  + results_foldername+ '/HeatCapacityT_simId={0}'.format(i)+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/HeatCapacityT_simId={0}.pgf'.format(i))

        # Residual entropy
        
        if RS:
            S = 0
            plt.figure(figsize=(12, 8), dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh] 
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                             t_h_S[i][tidmin:tidmax[i],hid],
                             '.', label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                            t_h_Smin[i][tidmin:tidmax[i],hid],
                            t_h_Smax[i][tidmin:tidmax[i],hid],\
                             alpha = alpha, color = col)
                plt.xlabel(r'Temperature $T$ ')
            plt.ylabel(r'$S$')
            plt.grid(which='both')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.savefig('./' + foldername  + results_foldername+ '/EntropyT_simId={0}'.format(i)+addsave+'.png')
            if pgf:
                plt.savefig('./' + foldername  + results_foldername+ '/EntropyT_simId={0}.pgf'.format(i))

        # Ground-state energy
        gs = kwargs.get('gs', False)
        if gs:
            print("gs check")
            r1 = [0, 0.5]
            E1 = [-2/3, -1/6]
            r2 = [0.5, 1]
            E2 = [-1/6, -1/3]
            r3 = [1, 4]
            E3 = [-1/3, -4+2/3]
            margin = [0.08, 0.08, 0.02, 0.1]
            plt.figure(figsize=(9, 5))
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            plt.plot(r1, E1, color = 'orange', label = r'$E - E_{NN}$ = $-\frac{2}{3}$ $J_2$ + $J_3$')
            plt.plot(r3, E3, color = 'red', label = r'$E - E_{NN} = \frac{2}{3}$ $J_2$ - $J_3$')
            plt.plot(r2, E2, '--', color = 'purple', label = r'$E- E_{NN} = -\frac{1}{3}J_3$')
            ratios = list()
            E = list()
            correction = list()
            
            print('Verifying that the low temperatures of file ', filenamelist[i], 'correspond to the ground state.')
            if J2[i] != 0:
                ratios.append(J3[i]/J2[i])
                E.append((t_h_MeanE[i][0] + 2/3 * J1[i])/J2[i])
                correction.append(t_h_varMeanE[i][0]/J2[i])
                #print(t_h_MeanE[i][0] + 2/3 * J1[i]+J3[i])
            else:
                print(t_h_MeanE[i][0] + 2/3 * J1[i]+J3[i])

            ratios = np.array(ratios)
            E = np.array(E)
            correction = np.array(correction)
            plt.plot(ratios, E, '.', label = r'Energy at $T = 0.05$ (N)')
            plt.fill_between(ratios , E - correction, E + correction, alpha = alpha, color = 'lightblue')
            #plt.plot(ratios, [0 for r in ratios], '.')
            plt.xlabel(r'$\frac{J_3}{J_2}$', size = 22)
            plt.ylabel(r'$\frac{E - E_{NN}}{J_2}$', size = 22)
            plt.grid(which='both')
            #plt.legend()
            plt.savefig('./' + foldername  + results_foldername + '/E(ratio)_simId={0}'.format(i)+addsave+'.png')
            if pgf:
                plt.savefig('./' + foldername  + results_foldername + '/E(ratio)_simId={0}.pgf'.format(i))
        
        gscheck = kwargs.get('gscheck', False)
        if gscheck:
            print('Verifying that the low temperatures of file ', filenamelist[i], 'correspond to the ground state.')
            print("Phase 1: ",testPhase(t_h_MeanE[i][0],  (-2/3 * J1[i] - 2/3 * J2[i] + J3[i])))
            print("Phase 2: ",testPhase(t_h_MeanE[i][0],  (-2/3 * J1[i] - 2/3 * J2[i] + 3 * J3[i])))
            print("Phase 3: ",testPhase(t_h_MeanE[i][0],  (-2/3 * J1[i] - 1/3 * J3[i])))
            print("Phase 4: ",testPhase(t_h_MeanE[i][0],   (-2/3 * J1[i] + 2/3 * J2[i] - J3[i])))
            print("Phase 5: ",testPhase(t_h_MeanE[i][0],   (-2/3 * J1[i] + 2 * J2[i] - J3[i])))
            print("Phase 6: ",testPhase(t_h_MeanE[i][0],   (-2/9 * J1[i] - 2/3 * J2[i] - 7/9 * J3[i])))
            print("Phase 7: ",testPhase(t_h_MeanE[i][0],  (-2/15 * J1[i] - 2/3 * J2[i] - J3[i])))
            print("Phase 8: ",testPhase(t_h_MeanE[i][0],   (2/3 * J1[i] - 2/3 * J2[i] - J3[i])))
            print("Phase 9: ",testPhase(t_h_MeanE[i][0],   (2/3 * J1[i] - 2/3 * J2[i] + 1/3 * J3[i])))
            print("Phase 10: ",testPhase(t_h_MeanE[i][0],   (6/7 * J1[i] - 2/7 * J2[i] - J3[i])))
            print("Phase 11: ",testPhase(t_h_MeanE[i][0],   (2 * J1[i] + 2 * J2[i] + 3 * J3[i])))


# In[ ]:


def BulkPlotsE(L, n, hid, tidmin, tidmax, temperatures_plots, foldername,
                results_foldername, filenamelist, t_h_MeanE, t_h_MeanEsq, t_h_varMeanE,
                t_h_varMeanEsq, t_h_VarE, t_h_ErrVarE, C, ErrC, J1, J2, J3, J4, 
                t_h_S, t_h_Smin, t_h_Smax, t_h_SE, t_h_SEmin, t_h_SEmax, **kwargs):
    
    margin = [0.15, 0.15, 0.02, 0.1]
    addsave = kwargs.get('addsave', "")
    alpha = kwargs.get('alpha', 0.2)
    figsize = kwargs.get('figsize', (6,4))
    markersize = kwargs.get('markersize', 10)
    put_legend = kwargs.get('put_legend', True)
    ncol = kwargs.get('ncol', 1)
    loc = kwargs.get('loc', 'best')
    fmts = kwargs.get('fmts', ['.' for i in range(n)])
    pltfill = kwargs.get('pltfill',[True for i in range(n)])
    
    plt.figure(figsize=figsize,dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                         t_h_MeanE[i][tidmin:tidmax[i]][:,hid],fmts[i], markersize=markersize,\
                          label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             (t_h_MeanE[i][tidmin:tidmax[i]][:,hid]
                              - np.sqrt(t_h_varMeanE[i][tidmin:tidmax[i]][:,hid])),
                             (t_h_MeanE[i][tidmin:tidmax[i]][:,hid]
                              + np.sqrt(t_h_varMeanE[i][tidmin:tidmax[i]][:,hid])),\
                             alpha=alpha)
    plt.xlabel(r'$T/|J_2|$')
    plt.ylabel(r'$E$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    
    plt.savefig('./' + foldername  + results_foldername+ '/Energy'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/Energy'+addsave+'.png')
            
    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_VarE[i][tidmin:tidmax[i]][:,hid],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         t_h_VarE[i][tidmin:tidmax[i]][:,hid]- t_h_ErrVarE[i][tidmin:tidmax[i]][:,hid],
                         t_h_VarE[i][tidmin:tidmax[i]][:,hid]+ t_h_ErrVarE[i][tidmin:tidmax[i]][:,hid],    
                         alpha = alpha)
    plt.xlabel(r'$T/|J_2|$ ')
    plt.ylabel(r'Var($E$)')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/VarianceE'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/VarianceE'+addsave+'.png')
        
    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     C[i][tidmin:tidmax[i]][:,hid],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         (C[i][tidmin:tidmax[i]][:,hid]
                          - ErrC[i][tidmin:tidmax[i]][:,hid]),
                         (C[i][tidmin:tidmax[i]][:,hid]
                          + ErrC[i][tidmin:tidmax[i]][:,hid]),
                         alpha = alpha)
    plt.xlabel(r'$T/|J_2|$ ')
    plt.ylabel(r'$c$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/SpecificHeat'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/SpecificHeat'+addsave+'.png')


    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])

    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     C[i][tidmin:tidmax[i]][:,hid] / temperatures_plots[i][tidmin:tidmax[i]],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         (C[i][tidmin:tidmax[i]][:,hid]
                          - ErrC[i][tidmin:tidmax[i]][:,hid]
                         )/temperatures_plots[i][tidmin:tidmax[i]],
                         (C[i][tidmin:tidmax[i]][:,hid]
                          + ErrC[i][tidmin:tidmax[i]][:,hid]
                         )/temperatures_plots[i][tidmin:tidmax[i]],\
                         alpha = alpha)
    plt.xlabel(r'$T/|J_2|$ ')
    plt.ylabel(r'$\frac{c}{T}$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/SpecificHeatOverT'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/SpecificHeatOverT'+addsave+'.png')



    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    if n == 1:
        plt.plot([temperatures_plots[0][tidmin],temperatures_plots[0][tidmax[0]-1]], [np.log(2), np.log(2)], '--')
    plt.gca().set_prop_cycle(None)
    for i in range(n):
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_S[i][tidmin:tidmax[i]][:,hid],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         t_h_Smin[i][tidmin:tidmax[i]][:,hid],
                         t_h_Smax[i][tidmin:tidmax[i]][:,hid],\
                         alpha = alpha)
    plt.xlabel(r'$T/|J_2|$ ')
    plt.ylabel(r'$S$')
    if n == 1:
        plt.ylim([0,0.7])
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy'+addsave+'.png')

    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    plt.gca().set_prop_cycle(None)
    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_SE[i][tidmin:tidmax[i]][:,hid],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         t_h_SEmin[i][tidmin:tidmax[i]][:,hid],
                         t_h_SEmax[i][tidmin:tidmax[i]][:,hid],\
                         alpha = alpha)

    plt.xlabel(r'$T/|J_2|$ ')
    plt.ylabel(r'$S$ from $E$')
    if n == 1:
        plt.ylim([0,0.7])
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/EntropyFromE'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/EntropyFromE'+addsave+'.png')
    
    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    plt.gca().set_prop_cycle(None)
    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.plot(temperatures_plots[i][tidmin:tidmax[i]],
                     C[i][tidmin:tidmax[i]][:,hid] / temperatures_plots[i][tidmin:tidmax[i]],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         (C[i][tidmin:tidmax[i]][:,hid]
                          - ErrC[i][tidmin:tidmax[i]][:,hid]
                         )/temperatures_plots[i][tidmin:tidmax[i]],
                         (C[i][tidmin:tidmax[i]][:,hid]
                          + ErrC[i][tidmin:tidmax[i]][:,hid]
                         )/temperatures_plots[i][tidmin:tidmax[i]],\
                         alpha = alpha)
    plt.xlabel(r'$T/|J_2|$ ')
    plt.xlim([0,None])
    plt.ylabel(r'$\frac{c}{T}$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/SpecificHeatOverT_Linear_'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/SpecificHeatOverT_Linear_'+addsave+'.png')


    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    plt.plot([0,temperatures_plots[0][tidmax[0]-1]], [np.log(2), np.log(2)], '--')
    
    plt.gca().set_prop_cycle(None)
    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.plot(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_S[i][tidmin:tidmax[i]][:,hid],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         t_h_Smin[i][tidmin:tidmax[i]][:,hid],
                         t_h_Smax[i][tidmin:tidmax[i]][:,hid],\
                         alpha = alpha)

    plt.xlabel(r'$T/|J_2|$ ')
    plt.xlim([0,temperatures_plots[0][tidmax[0]-1]])
    plt.ylim([0,0.7])
    plt.ylabel(r'$S$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy_Linear'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy_Linear'+addsave+'.png')
    
    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    plt.plot([0,temperatures_plots[0][tidmax[0]-1]], [np.log(2), np.log(2)], '--')
    plt.gca().set_prop_cycle(None)
    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.plot(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_SE[i][tidmin:tidmax[i]][:,hid],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                         t_h_SEmin[i][tidmin:tidmax[i]][:,hid],
                         t_h_SEmax[i][tidmin:tidmax[i]][:,hid],\
                         alpha = alpha)

    plt.xlabel(r'$T/|J_2|$ ')
    plt.xlim([0,temperatures_plots[0][tidmax[0]-1]])
    plt.ylim([0,0.7])
    plt.ylabel(r'$S$ from $E$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/EntropyFromE_Linear'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/EntropyFromE_Linear'+addsave+'.png')
    
    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    plt.gca().set_prop_cycle(None)
    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_SE[i][tidmin:tidmax[i]][:,hid]-t_h_S[i][tidmin:tidmax[i]][:,hid],
                     fmts[i], markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             t_h_SE[i][tidmin:tidmax[i]][:,hid]-t_h_Smin[i][tidmin:tidmax[i]][:,hid],
                             t_h_SE[i][tidmin:tidmax[i]][:,hid]-t_h_Smax[i][tidmin:tidmax[i]][:,hid],\
                         alpha =2*alpha)
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             t_h_SEmin[i][tidmin:tidmax[i]][:,hid]-t_h_S[i][tidmin:tidmax[i]][:,hid],
                             t_h_SEmax[i][tidmin:tidmax[i]][:,hid]-t_h_S[i][tidmin:tidmax[i]][:,hid],\
                         alpha = alpha)

    plt.xlabel(r'$T/|J_2|$ ')
    plt.ylabel(r'$S_E$ - $S$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy_Difference'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy_Difference'+addsave+'.png')
    
    plt.figure(figsize=figsize, dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    plt.gca().set_prop_cycle(None)
    for i in range(n):
        C[i] = np.array(C[i])
        ErrC[i] = np.array(ErrC[i])
        #col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_SE[i][tidmin:tidmax[i]][:,hid],
                     ".", markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             t_h_SEmin[i][tidmin:tidmax[i]][:,hid],
                             t_h_SEmax[i][tidmin:tidmax[i]][:,hid],\
                         alpha =2*alpha)
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                     t_h_S[i][tidmin:tidmax[i]][:,hid],
                     "x", markersize=markersize, label = r'$it$ = {0}'.format(i))
        if pltfill[i]:
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             t_h_Smin[i][tidmin:tidmax[i]][:,hid],
                             t_h_Smax[i][tidmin:tidmax[i]][:,hid],\
                         alpha = alpha)

    plt.xlabel(r'$T/|J_2|$ ')
    plt.ylabel(r'$S_E$ vs $S$')
    plt.grid(which = 'both', linestyle = '--', alpha = 0.3)
    if put_legend:
        plt.legend(loc= loc, ncol = ncol, framealpha=0.5)
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy_Comparison'+addsave+'.pdf')
    plt.savefig('./' + foldername  + results_foldername+ '/Entropy_Comparison'+addsave+'.png')


# In[ ]:


def BasicPlotsM(L, i, tidmin, tidmax, temperatures_plots, hfields_plots, foldername,
                results_foldername, filenamelist, t_h_MeanM, t_h_MeanMsq, 
                t_h_varMeanM, t_h_varMeanMsq, Chi, ErrChi, J1, J2, J3, J4, **kwargs):
    
    ploth = kwargs.get('ploth', False)
    pgf = kwargs.get('pgf', False)
    expm = kwargs.get('expm', 0)
    expmerr = kwargs.get('expmerr', 0)
    addsave = kwargs.get('addsave', "")
    alpha = kwargs.get('alpha', 0.2)
    TNvals = kwargs.get('TNvals', [])
    
    ## Magnetisation
    t_h_MeanM = np.array(t_h_MeanM)
    t_h_MeanMsq =  np.array(t_h_MeanMsq)
    t_h_varMeanM =  np.array(t_h_varMeanM)
    t_h_varMeanMsq =  np.array(t_h_varMeanMsq)
    Chi = np.array(Chi)
    ErrChi = np.array(ErrChi)
    
    
    #Magnetisation:
    margin = [0.08, 0.08, 0.02, 0.1]
    
    if ploth:
        #mt = len(temperatures_plots[i])
        mt = tidmax[i];
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for tid, t in enumerate(temperatures_plots[i]):
            if tid >= tidmin and tid <= tidmax[i]:
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i],
                                 t_h_MeanM[i][tid, :],'.',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i],
                                 (t_h_MeanM[i][tid,:]
                                  - np.sqrt(t_h_varMeanM[i][tid,:])),
                                 (t_h_MeanM[i][tid,:]
                                  + np.sqrt(t_h_varMeanM[i][tid,:])),\
                                 alpha = alpha, color = col)
                if expm != 0:
                    plt.fill_between([min(hfileds_plots[i]),max(hfields_plots[i])],[expm-expmerr,expm-expmerr],
                                     [expm+expmerr, expm+expmerr], alpha = alpha, label = r'$m$ - exp')

        if len(TNvals)>0:
            plt.plot(hfields_plots[i][0], TNvals[0], '*', label = "TN value", color = "blue")
        plt.xlabel(r'Magnetic field $h$')
        plt.ylabel(r'Magnetisation per site $m$')
        plt.grid(which='both')
        plt.legend(loc= 'best', framealpha = alpha)
        plt.title('Filename: '+filenamelist[i])
        plt.savefig('./' + foldername  + results_foldername
                    + '/h_M'+addsave+".png")
        if pgf:
            plt.savefig('./' + foldername  + results_foldername
                        + "/h_M"+addsave+".pgf")
    else:
        mh = len(hfields_plots[i])
        plt.figure(figsize=(12, 8), dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]], t_h_MeanM[i][tidmin:tidmax[i]][:,hid], 
                             '.',label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                                 (t_h_MeanM[i][tidmin:tidmax[i]][:,hid]
                                  - np.sqrt(t_h_varMeanM[i][tidmin:tidmax[i]][:,hid])),
                                 (t_h_MeanM[i][tidmin:tidmax[i]][:,hid]
                                  + np.sqrt(t_h_varMeanM[i][tidmin:tidmax[i]][:,hid])),\
                                 alpha = alpha, color = col)
                if expm != 0:
                    plt.fill_between([temperatures_plots[i][tidmin],temperatures_plots[i][tidmax[i]-1]],
                                     [expm-expmerr,expm-expmerr],
                                     [expm+expmerr, expm+expmerr], alpha = alpha, label = r'$m$ - exp')
        if len(TNvals)>0:
            plt.plot(temperatures_plots[i][tidmin], TNvals[0], '*', markersize = 10, label = "TN value", color = "blue")
        plt.xlabel(r'Temperature $T$ ')
        plt.ylabel('Magnetisation per site')
        plt.title('Filename: '+filenamelist[i])
        plt.grid(which='both')
        plt.legend(loc= 'upper right', framealpha = alpha)
        plt.savefig('./' + foldername  + results_foldername+ '/M'+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/M'+addsave+'.pgf')
    #Susceptibility
    if ploth:
        mt = tidmax[i];
        plt.figure(figsize=(12, 8),dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for tid, t in enumerate(temperatures_plots[i]):
            if tid >= tidmin and tid <= tidmax[i]:
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i],
                                 Chi[i][tid, :],'.',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i],
                                 (Chi[i][tid,:]
                                  - ErrChi[i][tid,:]),
                                 (Chi[i][tid,:]
                                  + ErrChi[i][tid,:]),\
                                 alpha = alpha, color = col)
        plt.xlabel(r'Magnetic field $h$')
        plt.ylabel('Susceptibility')
        plt.grid(which='both')
        plt.title('Filename: '+filenamelist[i])
        plt.savefig('./' + foldername  + results_foldername+ '/h_Susceptibility'+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/h_Susceptibility'+addsave+'.pgf')
    
    else:
        mh = len(hfields_plots[i])
        plt.figure(figsize=(12, 8), dpi=300)
        plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
        for hid, h in enumerate(hfields_plots[i]):
            col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                         Chi[i][tidmin:tidmax[i]][:,hid], '.',\
                         label = r'$h$ = {0}'.format(h), color = col)
            plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                             Chi[i][tidmin:tidmax[i]][:,hid]
                             - ErrChi[i][tidmin:tidmax[i]][:,hid], 
                             Chi[i][tidmin:tidmax[i]][:,hid]
                             + ErrChi[i][tidmin:tidmax[i]][:,hid],
                             alpha = alpha, color = col)
        plt.xlabel(r'Temperature $T$ ')
        plt.ylabel('Susceptibility')
        plt.grid(which='both')
        plt.title('Filename: '+filenamelist[i])
        plt.savefig('./' + foldername  + results_foldername+ '/Susceptibility'+addsave+'.png')
        if pgf:
            plt.savefig('./' + foldername  + results_foldername+ '/Susceptibility'+addsave+'.pgf')


# In[ ]:


def PrepPlot2DCorrelations(rid, n, t_h_MeanCorr,
                          t_h_errCorrEstim, t_h_MeanSi,
                          hfields_plots, temperatures_plots,\
                          ploth = False):
    if not ploth:
        corr = [[[] for h in hfields_plots[0]] for i in range(n)]
        errcorr = [[[] for h in hfields_plots[0]] for i in range(n)]
        maxerr = [[0 for h in hfields_plots[0]] for i in range(n)]
        for i in range(n):
            for hid, h in enumerate(hfields_plots[i]):
                corr[i][hid] = np.array(t_h_MeanCorr[i])[:,rid,hid]
                errcorr[i][hid] = np.sqrt(np.array(t_h_errCorrEstim[i])[:,rid,hid])
                maxerr[i][hid] = np.amax(np.abs(np.array(t_h_MeanSi[i])[:,rid,hid]))**2

    else:
        corr = [[[] for t in temperatures_plots[0]] for i in range(n)]
        errcorr = [[[] for t in temperatures_plots[0]] for i in range(n)]
        maxerr = [[0 for t in temperatures_plots[0]] for i in range(n)]
        for i in range(n):
            for tid, t in enumerate(temperatures_plots[i]):
                corr[i][tid] = np.array(t_h_MeanCorr[i])[:,tid,rid,:]
                errcorr[i][tid] = np.sqrt(np.array(t_h_errCorrEstim[i])[:,tid,rid,:])
                maxerr[i][tid] = np.amax(np.abs(np.array(t_h_MeanSi[i])[:,tid,rid]))**2

    return corr, errcorr, maxerr


# In[ ]:


def BasicPlotsCorrelations2D(foldername, results_foldername, rid,
                             n, L, corr, errcorr, t_h_MeanSi,
                             hfields_plots, temperatures_plots,\
                             ploth = False, pgf = False):
    if not ploth:
        matplotlib.rcParams.update({'font.size': 6})
        for i in range(n):
            a =1
            for hid, h in enumerate(hfields_plots[i]):
                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][hid][0], L[i], a)
                plt.title('L = {0}; h = {1}'.format(L[i], h))
                plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations0_L{0}_h={1}.png'.format(L[i], h))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations0_L{0}_h={1}.pgf'.format(L[i],h))
                plt.show()


                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][hid][1], L[i], a)
                plt.title('L = {0}; h = {1}'.format(L[i], h))
                plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations1_L{0}_h={1}.png'.format(L[i],h))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername +                            '/Correlations1_L{0}_h={1}.pgf'.format(L[i],h))
                plt.show()

                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][hid][2], L[i], a)
                plt.title('L = {0}; h = {1}'.format(L[i], h))
                plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations_L{0}_h={1}.png'.format(L[i],h))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations_L{0}_h={1}.pgf'.format(L[i],h))
                plt.show()
                
        avgsi =  [[[] for h in hfields_plots[0]]
                  for i in range(n)]
        for i in range(n):
            for hid,h in enumerate(hfields_plots[i]):
                avgsi[i][hid] = np.array(t_h_MeanSi[i])[rid,hid,:]
        for i in range(n):
            a = 1
            for hid, h in enumerate(hfields_plots[i]):
                plt.figure(dpi=300)
                kdraw.plot_function_kag(avgsi[i][hid], L[i], a)
                plt.title('L = {0}; h = {1}'.format(L[i], h))
                #plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Spinaverage_L{0}_h={1}.png'.format(L[i],h))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername+                            '/Spinaverage_L{0}_h={1}.pgf'.format(L[i],h))
                plt.show()
    
    else:
        matplotlib.rcParams.update({'font.size': 6})
        for i in range(n):
            a =1
            for tid, t in enumerate(temperatures_plots[i]):
                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][tid][0], L[i], a)
                plt.title('L = {0}; t = {1}'.format(L[i], t))
                plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations0_L{0}_t={1}.png'.format(L[i],t))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations0_L{0}_t={1}.pgf'.format(L[i],t))
                plt.show()


                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][tid][1], L[i], a)
                plt.title('L = {0}; t = {1}'.format(L[i], t))
                plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations1_L{0}_t={1}.png'.format(L[i],t))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations1_L{0}_t={1}.pgf'.format(L[i],t))
                plt.show()

                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][tid][2], L[i], a)
                plt.title('L = {0}; t = {1}'.format(L[i], t))
                plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations2_L{0}_t={1}.png'.format(L[i],t))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername+                            '/Correlations2_L{0}_t={1}.pgf'.format(L[i],t))
                plt.show()
        
        avgsi =  [[[] for t in temperatures_plots[0]]
                  for i in range(n)]
        for i in range(n):
            for tid, t in enumerate(temperatures_plots[i]):
                avgsi[i][tid] = np.array(t_h_MeanSi[i])[tid,rid,:]
        for i in range(n):
            a = 1
            for tid, t in enumerate(temperatures_plots[i]):
                plt.figure(dpi=300)
                kdraw.plot_function_kag(avgsi[i][tid], L[i], a)
                plt.title('L = {0}; T = {1}'.format(L[i], t))
                #plt.clim(-1,1)
                plt.savefig('./' + foldername  +                            results_foldername+                            '/Spinaverage_L{0}_t={1}.png'.format(L[i],t))
                if pgf:
                    plt.savefig('./' + foldername  +                            results_foldername+                            '/Spinaverage_L{0}_t={1}.pgf'.format(L[i],t))
                plt.show()
                


# In[ ]:


def PlotStrctFact(StrctFact, foldername, results_foldername, tid,
                  hid,L, i, hfields_plots, temperatures_plots,
                  vmindiag = None, vmaxdiag = None,
                  vminoff = None, vmaxoff = None, **kwargs):
    
    
    size = (170/L[i])**2
    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,0,0]),
                                   L[i], 2, s = size, vmin = vmindiag,
                                   vmax = vmaxdiag, **kwargs)
    plt.title('L = {0}; h = {1}; SF 00'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername  + results_foldername+                '/SF00_L={0}_h={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,1,1]),
                                   L[i], 2, s = size, vmin = vmindiag,
                                   vmax = vmaxdiag, **kwargs)
    plt.title('L = {0}; h = {1}; SF 11'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername  + results_foldername+                '/SF11_L={0}_h={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,2,2]),
                                   L[i], 2, s = size, vmin = vmindiag,
                                   vmax = vmaxdiag, **kwargs)
    plt.title('L = {0}; h = {1}; SF 22'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername  + results_foldername+                '/SF22_L={0}_h={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,0,1]
                                           +StrctFact[:,1,0])/2,
                                   L[i], 2, s = size, vmin = vminoff,
                                   vmax = vmaxoff, **kwargs)
    plt.title('L = {0}; h = {1}; SF 01 + 10'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername  + results_foldername+                '/SF01_L={0}_h={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,0,2]
                                           +StrctFact[:,2,0])/2,
                                   L[i], 2, s = size, vmin = vminoff,
                                   vmax = vmaxoff, **kwargs)
    plt.title('L = {0}; h = {1}; SF 02 + 20'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername  + results_foldername+                '/SF02_L={0}_h={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,1,2]
                                           +StrctFact[:,2,1])/2,
                                   L[i], 2, s = size, vmin = vminoff,
                                   vmax = vmaxoff, **kwargs)
    plt.title('L = {0}; h = {1}; SF 12 + 21'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername  + results_foldername+                '/SF12_L={0}_h={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))


# In[ ]:


def dist_corr(L, findex, corr, errcorr,distmax, srefs, nnlists):
    # for now, doing this:
    
    #distances, distances_spins, NNList, s_pos = kf.NearestNeighboursLists(L, distmax, srefs)
    # instead, consider using the same NNlist as given to FirstCorrelations
    
    
    #C = [[0 for i in range(len(NNList[0]))] for j in range(len(srefs))]
    #ErrC = [[0 for i in range(len(NNList[0]))] for j in range(len(srefs))]
    #for j in range(len(srefs)):
    #    for i in range(len(NNList[0])):
    #        Corrji = 0
    #        ErrCorrji = 0
    #        count = 0
    #        for pair in NNList[j][i]:
    #            if srefs[j] == pair[0]:
    #                count += 1
    #                Corrji += corr[findex][j][pair[1]]
    #                ErrCorrji += errcorr[findex][j][pair[1]]
    #        if count == 0:
    #            print("NNList[", j, "][", i, "] = ", NNList[j][i])
    #        Corrji = Corrji/count
    #        ErrCorrji = ErrCorrji/count
    #        
    #        C[j][i] = Corrji
    #        ErrC[j][i]= ErrCorrji
    #    C[j] = np.array(C[j])
    #    ErrC[j] = np.array(ErrC[j])
    
    assert len(srefs) == 3
    C = np.zeros([len(srefs), len(nnlists)])
    ErrC = np.zeros([len(srefs), len(nnlists)])
    
    
    
    for j in range(len(srefs)):
        #print('sref', srefs[j])
        for i in range(len(nnlists)):
            nns = np.array(nnlists[i]);
            ids0 = np.where(nns[:,0] == srefs[j]);
            ids1 = np.where(nns[:,1] == srefs[j]);
            
            numberneis = ids0[0].shape[0] + ids1[0].shape[0];
            #print(i)
            #print('ids', ids0[0], " and ", ids1[0])
            #print('nns', nns[ids0[0]][:,1], " and ", nns[ids1[0]][:,0])
            #print('corr shape', corr[findex].shape)
            #print('correlations',  corr[findex][j, nns[ids0[0]][:,1]], " and ", corr[findex][j, nns[ids1[0]][:,0]])
            ##
            #print('correlations - sum',  corr[findex][j, nns[ids0[0]][:,1]].sum()+corr[findex][j, nns[ids1[0]][:,0]].sum())
            #
            assert numberneis > 0
            C[j,i] += corr[findex][j, nns[ids0[0]][:,1]].sum()/numberneis;
            C[j,i] += corr[findex][j, nns[ids1[0]][:,0]].sum()/numberneis;
            
            ErrC[j,i] += errcorr[findex][j, nns[ids0[0]][:,1]].sum()/numberneis;
            ErrC[j,i] += errcorr[findex][j, nns[ids1[0]][:,0]].sum()/numberneis;
            
            
    C =C.sum(0)/3
    ErrC = ErrC.sum(0)/3
    
    return C, ErrC
        


# In[ ]:


def PlotFirstCorrelations(i, L, foldername, results_foldername,hfields_plots, temperatures_plots,
                         t_h_MeanCorr, t_h_errCorrEstim, srefs, distmax = 3.5, ploth = False,
                          tmin = 0, setyticks = None, addtitle = "", addsave = "", save = True, log = True,
                          figsize=(11,9), dpi = 200,
                          **kwargs):

    distmax = min(3.5, distmax)
    nlistnames = ['$c_1$', '$c_2$', '$c_{3||}$', '$c_{3\star}$', '$c_{4}$', '$c_{5}$', '$c_{6}$', '$c_{6\star}$']
    
    rmmag = kwargs.get('rmmag', False)
    
    plotfirst = kwargs.get('plotFirst', False)
    if plotfirst: 
        t_h_MeanFc = kwargs.get('t_h_MeanFc')
    if not ploth:
        #spin table and dictionary
        (s_ijl, ijl_s) = kf.createspinsitetable(L[i])
        nnlists = [dw.NNpairs(ijl_s, s_ijl, L[i]), dw.NN2pairs(ijl_s, s_ijl, L[i]),
                   dw.NN3parpairs(ijl_s, s_ijl, L[i]), dw.NN3starpairs(ijl_s, s_ijl, L[i])]
        for hid, h in enumerate(hfields_plots[i]):
            fig, ax = plt.subplots(figsize = figsize,dpi=dpi)
            ax.set_xscale("log")
            if len(hfields_plots[i])!=1:
                plt.title('First few neighbours correlations' + addtitle+',                h = {0}'.format(h))
            else:
                plt.title('First few neighbours correlations' + addtitle)

            fmts = ['.','x','v','*','o','^','s']
            length = len(temperatures_plots[i])
            for t in range(1,length):

                corr = [np.array(t_h_MeanCorr[i])[:,t,hid,:]]
                #print(corr[0].shape)
                errcorr =                [np.sqrt(np.array(t_h_errCorrEstim[i])[:,t,hid])]

                (rescorr, reserrcorr) =                dist_corr(L[i],0 ,corr, errcorr, distmax, srefs[i], nnlists)

                if t == 1:
                    print(rescorr)

                plt.gca().set_prop_cycle(None)
                alpha = 0.5
                for nei in range(0,len(rescorr)):
                    if t == 1:
                        plt.errorbar(temperatures_plots[i][t],
                                     rescorr[nei],
                                     reserrcorr[nei],\
                                     fmt = fmts[nei],\
                                     label =\
                                     r'{0}'.format(nlistnames[nei]),\
                                     alpha = alpha)
                    else:
                        plt.errorbar(temperatures_plots[i][t],
                                     rescorr[nei],
                                     reserrcorr[nei],\
                                     fmt = fmts[nei],\
                                     alpha = alpha)

            if plotfirst:
                plt.gca().set_prop_cycle(None)
                plt.semilogx(temperatures_plots[i],t_h_MeanFc[i][:,hid,0],'.')
                plt.semilogx(temperatures_plots[i],t_h_MeanFc[i][:,hid,1],'.')
                plt.semilogx(temperatures_plots[i],t_h_MeanFc[i][:,hid,2],'.')
                plt.semilogx(temperatures_plots[i],t_h_MeanFc[i][:,hid,3],'.')
            plt.xlabel(r'$T/J_1$')
            if rmmag:
                plt.ylabel(r'$<\sigma_i \sigma_j> - <\sigma_i> <\sigma_j> $')
            else:
                plt.ylabel(r'$<\sigma_i \sigma_j>$')

            plt.grid(which='both')
            plt.legend(loc = 'best')
            if not plotfirst:
                plt.savefig('./' + foldername  +                            results_foldername+                            '/FewCorrelations_L={0}_h={1}_simId={2}'+addsave+'.png'.format(L[i],h,i))
            else:
                plt.savefig('./' + foldername  +                            results_foldername+                            '/FewCorrelationsComparison_L={0}_h={1}_simId={2}'+addsave+'.png'.format(L[i],h,i))
    else:
        #spin table and dictionary
        (s_ijl, ijl_s) = kf.createspinsitetable(L[i])
        nnlists = [dw.NNpairs(ijl_s, s_ijl, L[i]), dw.NN2pairs(ijl_s, s_ijl, L[i]),
                   dw.NN3parpairs(ijl_s, s_ijl, L[i]), dw.NN3starpairs(ijl_s, s_ijl, L[i])]
        for tid, t in enumerate(temperatures_plots[i]):
            fig, ax = plt.subplots(figsize = figsize, dpi=dpi)
            plt.title('First few neighbours correlations' + addtitle+',            t = {0}'.format(t))
            fmts = ['.','x','v','*','o','^','s']
            length = len(hfields_plots[i])
            for hid in range(1,length):

                corr = [np.array(t_h_MeanCorr[i])[:,tid,hid,:]]
                errcorr =                [np.sqrt(np.array(t_h_errCorrEstim[i])[:,tid,hid])]

                (rescorr, reserrcorr) =                dist_corr(L[i],0 ,corr, errcorr, distmax, srefs[i], nnlists)

                if hid == 1:
                    print(rescorr)

                plt.gca().set_prop_cycle(None)
                alpha = 0.5
                for nei in range(0,len(rescorr)):
                    if hid == 1:
                        plt.errorbar(hfields_plots[i][hid],
                                     rescorr[nei],
                                     reserrcorr[nei],\
                                     fmt = fmt,\
                                     label =\
                                     r'{0}'.format(nlistnames[nei]),\
                                     alpha = alpha)
                    else:
                        plt.errorbar(hfields_plots[i][hid],
                                     rescorr[nei],
                                     reserrcorr[nei],\
                                     fmt = fmt,\
                                     alpha = alpha)

            plt.xlabel(r'$h/J_1$')
            if rmmag:
                plt.ylabel(r'$<\sigma_i \sigma_j> - <\sigma_i> <\sigma_j> $')
            else:
                plt.ylabel(r'$<\sigma_i \sigma_j>$')
            plt.grid(which='both')    
            plt.legend(loc = 'best')
            plt.savefig('./' + foldername  +                        results_foldername+                        'FewCorrelations_L={0}_t={1}_simId={2}'+addsave+'.png'.format(L[i],t,i))

