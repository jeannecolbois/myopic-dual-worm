
# coding: utf-8

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import KagomeFunctions as kf # "library" allowing to work on Kagome
import DualwormFunctions as dw
import KagomeDrawing as kdraw
import KagomeFT as kft


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
    couplings = {'J1':0, 'J2' : 0, 'J3' : 1.0, 'J3st':1.0}
    hamiltonian = dw.Hamiltonian(couplings, d_ijl, ijl_d, L)
    EJ3 = dw.compute_energy(hamiltonian, state, latsize)
    
    # J4 #
    couplings = {'J1':0, 'J2' : 0, 'J3' : 0, 'J3st':0, 'J4': 1.0}
    hamiltonian = dw.Hamiltonian(couplings, d_ijl, ijl_d, L)
    EJ4 = dw.compute_energy(hamiltonian, state, latsize)
    
    config = {'J1':EJ1, 'J2': EJ2, 'J3':EJ3, 'J4': EJ4}
    print('J1 :', EJ1, '\nJ2 :', EJ2, '\nJ3 :', EJ3, '\nJ4 :', EJ4)
    
    return config


# In[ ]:


def LoadParameters(foldername, filenamelist):
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
    
    for nf, filename in enumerate(filenamelist):
        [L[nf], numsites[nf], J1[nf], J2[nf], J3[nf], J3st[nf], J4[nf], nb[nf], 
         num_in_bin[nf], temperatures[nf], nt[nf], stat_temps[nf], temperatures_plots[nf],
         hfields[nf], nh[nf], stat_hfields[nf], hfields_plots[nf],
         listfunctions[nf], sref[nf]] = LoadParametersFromFile(foldername, filename)
    
    return L, numsites, J1, J2, J3, J3st, J4, nb, num_in_bin, temperatures, nt,             stat_temps, temperatures_plots, hfields, nh,             stat_hfields, hfields_plots, listfunctions, sref


# In[ ]:


def LoadParametersFromFile(foldername, filename):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) #save the parameters
    L = backup.params.L
    
    #spin table and dictionary
    (s_ijl, ijl_s) = kf.createspinsitetable(L)
    numsites = len(s_ijl)
    
    #couplings
    J1 = backup.params.J1
    J2 = backup.params.J2
    J3 = backup.params.J3
    J3st = backup.params.J3st
    J4 = backup.params.J4
    
    nb = backup.params.nb
    num_in_bin = backup.params.num_in_bin
    
    temperatures = backup.params.temperatures.tolist()
    nt = backup.params.nt
    stat_temps = backup.params.stat_temps
    temperatures_plots = [temperatures[t] for t in stat_temps]
    
    hfields = backup.params.hfields.tolist()
    nh = backup.params.nh
    stat_hfields = backup.params.stat_hfields
    hfields_plots = [hfields[h] for h in stat_hfields]
    
    listfunctions = backup.results.namefunctions
    
    #reference spins
    s0 = ijl_s[L, L, 0]
    s1 = ijl_s[L, L, 1]
    s2 = ijl_s[L, L, 2]
    
    sref = [s0, s1, s2]
    f.close()
    
    return L, numsites, J1, J2, J3, J3st, J4, nb, num_in_bin, temperatures, nt,             stat_temps, temperatures_plots, hfields, nh,             stat_hfields, hfields_plots, listfunctions, sref


# In[ ]:


def ExtractStatistics(idfunc, statstable, nb, stat_temps, stat_hfields, sq = 0, **kwargs):
    '''
        This function gets the statistics from a file and
        computes the expectation values and variances of 
        the operator corresponding to idfunc
        
        sq = 0 -> not square stats
        sq = 1 -> square stats
    '''
    stattuple = statstable[idfunc];
    t_h_meanfunc = np.array(stattuple[sq]).sum(2)/nb
    
    binning = kwargs.get('binning', False)

    t_h_varmeanfunc = [[0 for h in stat_hfields] for t in stat_temps]
    for resid, t in enumerate(stat_temps):
        for reshid, h in enumerate(stat_hfields):
            for b in range(nb):
                t_h_varmeanfunc[resid][reshid] += ((stattuple[sq][resid][reshid][b] - t_h_meanfunc[resid][reshid]) ** 2)/(nb * (nb - 1))
    if binning:
        print('binning!')
        Binning(t_h_meanfunc,t_h_varmeanfunc, stattuple[sq], nb,
                                stat_temps, stat_hfields, **kwargs)
        
    return t_h_meanfunc, t_h_varmeanfunc


# In[ ]:


def Binning(t_h_mean, t_h_varmean, stattuple, nb, stat_temps,stat_hfields, **kwargs):
    '''
        This function implements a binning analysis
    '''
    ### NAIVE IMPLEMENTATION
    nblist = []
    nbb = nb
    while nbb >= 15:
        nblist.append(nbb)
        nbb = nbb//2
        
    print(" bins list for binning: ", nblist)
        
    t_h_vars = [[[] for reshid in range(len(stat_hfields))] for resid in range(len(stat_temps))]
    for resid, t in enumerate(stat_temps):
        for reshid, h in enumerate(stat_hfields):
            var = []
            for l,nbb in enumerate(nblist):
                avg = np.array(stattuple[resid][0:(2**l)]).sum(0)/(2**l)
                varl=((avg - t_h_mean[resid])**2)/(nbb*(nbb-1))
                for b in range(1,nbb):
                    avg = np.array(stattuple[resid][(2**l)*b:(2**l)*(b+1)]).sum(0)/(2**l)
                    varl+=((avg - t_h_mean[resid])**2)/(nbb*(nbb-1))
                if len(varl.shape) == 0:
                    var.append(varl)
                else:
                    var.append(np.max(varl))
                if resid == 0:
                    print(nbb, " --- ", var[l])
            
            t_h_vars[resid][reshid] = var
            
    plzplot = kwargs.get('plzplot', False)
    plotmin = kwargs.get('plotmin', 0)
    plotmax = kwargs.get('plotmax', 10)
    plothmin = kwargs.get('plothmin', 0)
    plothmax = kwargs.get('plothmax', 10)
    if plzplot:
        print('plotting!')
        plt.figure(figsize=(12, 8),dpi=300)
        minplt = max(0, plotmin)
        maxplt = min(plotmax, len(stat_temps))
        minhplt = max(0, plothmin)
        maxhplt = min(plothmax, len(stat_hfields))
        for reshid, h in enumerate(stat_hfields[minhplt:maxhplt]):
            for resid, t in enumerate(stat_temps[minplt:maxplt]):
                plt.plot(range(len(t_h_vars[resid][reshid])), t_h_vars[resid][reshid], '.-', label = 't = {0}'.format(t))
            plt.title('h = {0}'.format(h))
            plt.legend()
            plt.show()
    t_h_varmean = [[max(var) for var in h_vars] for h_vars in t_h_vars]


# In[ ]:


def LoadSwaps(foldername, filenamelist, nb, num_in_bin, nh, nt):
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
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f)
    
    nrps = backup.params.nrps
    nsms = nb*num_in_bin
    measperiod = backup.params.measperiod
    
    swapst_th = backup.results.swapst_th
    swapsh_th = backup.results.swapsh_th
    
    swapst = 4*np.array(backup.results.swapst)/(nsms*nrps*measperiod*nh)
    swapsh = 4*np.array(backup.results.swapsh)/(nsms*nrps*measperiod*nt)
    f.close()
    return swapst_th, swapsh_th, swapst, swapsh


# In[ ]:


def LoadEnergy(foldername, filenamelist, numsites, nb, stat_temps, temperatures, stat_hfields, listfunctions, **kwargs):
    n = len(filenamelist)
    
    t_MeanE = [[] for _ in range(n)]
    t_MeanEsq = [[] for _ in range(n)]
    t_varMeanE = [[] for _ in range(n)]
    t_varMeanEsq = [[] for _ in range(n)]
    C = [[] for _ in range(n)]
    ErrC = [[] for _ in range(n)]

    for nf, filename in enumerate(filenamelist):
        if 'Energy' in listfunctions[nf]:
            idfunc = listfunctions[nf].index('Energy')
            [t_MeanE[nf], t_MeanEsq[nf], t_varMeanE[nf], t_varMeanEsq[nf], C[nf], ErrC[nf]] =                 LoadEnergyFromFile(foldername, filename, numsites[nf], nb[nf], stat_temps[nf],
                                   temperatures[nf], stat_hfields[nf], idfunc, **kwargs)
        else:
            [t_MeanE[nf], t_MeanEsq[nf], t_varMeanE[nf], t_varMeanEsq[nf], C[nf], ErrC[nf]] = [[],[],[],[],[],[]]
        
    return t_MeanE, t_MeanEsq, t_varMeanE, t_varMeanEsq, C, ErrC


# In[ ]:


def LoadEnergyFromFile(foldername, filename, numsites, nb, stat_temps,
                       temperatures, stat_hfields, idfunc, **kwargs):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) 
    
    statstable = backup.results.statstable
    
    t_h_MeanE, t_h_varMeanE =    ExtractStatistics(idfunc, statstable, nb, 
                      stat_temps,stat_hfields, **kwargs)
    t_h_MeanEsq, t_h_varMeanEsq =    ExtractStatistics(idfunc, statstable, nb,
                      stat_temps, stat_hfields, sq = 1, **kwargs)
    
    C = []
    for resid, t in enumerate(stat_temps):
        Ch = []
        T = temperatures[t]
        for reshid, h in enumerate(stat_hfields):
            Ch.append(numsites * (t_h_MeanEsq[resid][reshid] -
                                  t_h_MeanE[resid][reshid] ** 2) / T ** 2)
        C.append(Ch)
    # to compute the error on C, we need to compute sqrt(<C^2> - <C>^2)
    # where "<>" stands for the average over all the bins
    # i.e. <C> = 1/nb * sum_b C_b where C_b is the value of C over the bin b
    # Note that C_b = N/T^2 * (<E^2>_b - <E>_b ^2)
    # where <>_b stands for the average over bin b

    tb_E = statstable[idfunc][0]
    tb_Esq = statstable[idfunc][1]

    ErrC = []
    for resid, t in enumerate(stat_temps):
        ErrCh = []
        for reshid, h in enumerate(stat_hfields):
            T = temperatures[t]
            Mean_VarE = 0
            Mean_VarE_Sq = 0
            for b in range(nb):
                    Mean_VarE += (tb_Esq[resid][reshid][b] - 
                                  tb_E[resid][reshid][b] ** 2)/nb
                    Mean_VarE_Sq += ((tb_Esq[resid][reshid][b] -
                                      tb_E[resid][reshid][b] ** 2) ** 2)/nb
            if (Mean_VarE_Sq - Mean_VarE ** 2 >= 0) :
                ErrCh.append(numsites / (T ** 2) * np.sqrt(Mean_VarE_Sq 
                                                           - Mean_VarE ** 2))
            else:
                assert(Mean_VarE_Sq - Mean_VarE ** 2 >= -1e-15)
                ErrCh.append(0)
        ErrC.append(ErrCh)
            
    f.close()
    
    return t_h_MeanE, t_h_MeanEsq, t_h_varMeanE, t_h_varMeanEsq, C, ErrC


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
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) 
    
    statstable = backup.results.statstable
    
    t_h_MeanM, t_h_varMeanM =    ExtractStatistics(idfunc, statstable, nb, stat_temps,
                      stat_hfields, **kwargs)
    t_h_MeanMsq, t_h_varMeanMsq =    ExtractStatistics(idfunc, statstable, nb, stat_temps,
                      stat_hfields, sq = 1, **kwargs)
    Chi = []
    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        Chih = []
        for reshid, h in enumerate(stat_hfields):
            Chih.append(numsites * ( t_h_MeanMsq[resid][reshid] -  t_h_MeanM[resid][reshid] ** 2) / T)
        Chi.append(Chih)
        
    tb_M = statstable[idfunc][0]
    tb_Msq = statstable[idfunc][1]

    
    ErrChi = []
    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        ErrChih = []
        for reshid, h in enumerate(stat_hfields):
            Mean_VarM = 0
            Mean_VarM_Sq = 0
            for b in range(nb):
                    Mean_VarM += (tb_Msq[resid][reshid][b] - tb_M[resid][reshid][b] ** 2)/nb
                    Mean_VarM_Sq += ((tb_Msq[resid][reshid][b] - tb_M[resid][reshid][b] ** 2) ** 2)/nb
            if Mean_VarM_Sq - Mean_VarM ** 2 >= 0 :
                ErrChih.append(numsites / T * np.sqrt(Mean_VarM_Sq - Mean_VarM ** 2))  
            else:
                assert(Mean_VarM_Sq - Mean_VarM ** 2 >= -1e-15)
                ErrChih.append(0)
        ErrChi.append(ErrChih)
            
    f.close()
    
    return  t_h_MeanM,  t_h_MeanMsq, t_h_varMeanM, t_h_varMeanMsq, Chi, ErrChi


# In[ ]:


def LoadCentralCorrelations(foldername, filenamelist, listfunctions, sref, stat_temps, stat_hfields, nb, **kwargs):
    n = len(filenamelist)
    
    ## "Correlations" <sisj>
    t_h_MeanSs = [[] for _ in range(n)]
    t_h_varMeanSs = [[] for _ in range(n)]
    
    ## Local spin average
    t_h_MeanSi = [[] for _ in range(n)]
    t_h_varMeanSi = [[] for _ in range(n)]
    
    ## Correlations
    t_h_MeanCorr = [[] for _ in range(n)]
    t_h_errCorrEstim = [[] for _ in range(n)]
    for nf, filename in enumerate(filenamelist):
        if ('Central_Correlations' in listfunctions[nf] 
            and 'Si' in listfunctions[nf]):
            # This will be improved when we will work with a more
            # general way of handling the correlations
            idfunc = listfunctions[nf].index('Central_Correlations')
            idfuncsi = listfunctions[nf].index('Si')

            [t_h_MeanSs[nf], t_h_varMeanSs[nf], t_h_MeanSi[nf],
             t_h_varMeanSi[nf], t_h_MeanCorr[nf],t_h_errCorrEstim[nf]] =\
            LoadCorrelationsFromFile(foldername, filename, idfunc,
                                     idfuncsi, sref[nf], stat_temps[nf],
                                     stat_hfields[nf], nb[nf], **kwargs)
        else:
            [t_h_MeanSs[nf], t_h_varMeanSs[nf], t_h_MeanSi[nf], t_h_varMeanSi[nf], t_h_MeanCorr[nf], 
             t_h_errCorrEstim[nf]] = [[],[],[],[],[]]
    return t_h_MeanSs, t_h_varMeanSs, t_h_MeanSi, t_h_varMeanSi, t_h_MeanCorr, t_h_errCorrEstim


# In[ ]:


def LoadCorrelationsFromFile(foldername, filename, idfunc, idfuncsi, sref, stat_temps, stat_hfields, nb, **kwargs):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) 
    
    statstable = backup.results.statstable
    
    
    # Averages and corresponding variances
    t_h_MeanSi, t_h_varMeanSi =    ExtractStatistics(idfuncsi, statstable, nb, stat_temps,
                      stat_hfields, **kwargs)
    
    t_h_MeanSs, t_h_varMeanSs =    ExtractStatistics(idfunc, statstable, nb, stat_temps,
                      stat_hfields, **kwargs)
    
    
    t_h_MeanCorr = []
    for i in range(len(sref)):
        column = t_h_MeanSi[:, :, sref[i]]
        column = column[:,:,np.newaxis]
        t_h_MeanCorr.append(t_h_MeanSs[:,:,i,:] - t_h_MeanSi*column) #<si sj> - <si> <sj> for j in lattice. /!\ this is ELEMENTWISE
        
    # Estimating the error on <si sj> - <si><sj>
    t_h_errCorrEstim = CorrelErrorEstimator(statstable, idfunc, idfuncsi, sref, nb)   
 
    f.close()
    return t_h_MeanSs, t_h_varMeanSs, t_h_MeanSi, t_h_varMeanSi, t_h_MeanCorr, t_h_errCorrEstim


# In[ ]:


def CorrelErrorEstimator(statstable, idfunc, idfuncsi, sref, nb):
    
    t_h_b_sisj = np.array(statstable[idfunc][0]) # <s0sj>_b (t)
    (ntm, nhm, nb, nrefs, nsites) = t_h_b_sisj.shape #getting the system size
    
    t_h_b_sj = np.array(statstable[idfuncsi][0]) # <si>_b (t)
    t_h_b_s0 = t_h_b_sj[:,:,:,sref]
    
    t_h_b_gamma = [[] for _  in range(nrefs)]
    t_h_gamma = [np.zeros((ntm, nhm, nsites)) for _ in range(nrefs)]
    for i in range(nrefs):
        t_h_b_s0i = t_h_b_s0[:,:,:,i]
        t_h_b_s0i = t_h_b_s0i[:,:,:,np.newaxis]
        
        for b in range(nb):
            t_h_b_gamma[i].append(t_h_b_sisj[:,:,b,i,:] - t_h_b_s0i[:,:,b]*t_h_b_sj[:,:,b,:])
            t_h_gamma[i] += t_h_b_gamma[i][b]
        t_h_gamma[i] = t_h_gamma[i]/nb
    
    
    t_h_vargamma = [np.zeros((ntm, nhm, nsites)) for _ in range(nrefs)]
    for i in range(nrefs):
        for b in range(nb):
            t_h_vargamma[i] += np.power((t_h_b_gamma[i][b] - t_h_gamma[i]),2)
        t_h_vargamma[i] = t_h_vargamma[i]/(nb*(nb-1))

    return t_h_vargamma


# In[ ]:


def LoadSi(foldername, filenamelist, listfunctions, **kwargs):
    n = len(filenamelist)
    
    t_MeanSi = [[] for _ in range(n)]
    t_varMeanSi = [[] for _ in range(n)]

    for nf, filename in enumerate(filenamelist):
        if 'Si' in listfunctions[nf]:
            idfunc = listfunctions[nf].index('Si')
            [t_MeanSi[nf], t_varMeanSi[nf]] = LoadSiFromFile(foldername, filename, idfunc,stat_temps[nf], **kwargs)
        else:
            [t_MeanSi[nf], t_varMeanSi[nf]] = [[],[]]
            
    return t_MeanSi, t_varMeanSi


# In[ ]:


def LoadSiFromFile(foldername, filename, idfunc, stat_temps, **kwargs):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) 
    
    statstable = backup.results.statstable
    
    t_MeanSi, t_varMeanSi = ExtractStatistics(idfuncsi, statstable, nb, stat_temps, **kwargs)
    
    f.close()
    
    return t_MeanSi, t_varMeanSi
    


# In[ ]:


def SwapsAnalysis(L, n, tidmin, tidmax, temperatures, hfields, foldername, results_foldername, swapst, swapsh):
    for i in range(n):
        plt.figure()
        plt.loglog(temperatures[i][tidmin:tidmax[i]-1], swapst[i][tidmin:tidmax[i]-1], '.-', color = 'green')
        plt.xlabel('Temperature')
        plt.ylabel('Ratio of swaps')
        plt.title('Ratio of swaps as a function of the temperature')
        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/NumberSwapsTemperature_L={0}_various-nsms.png'.format(L[i]))
        
        nh = len(hfields[i])
        plt.figure()
        plt.semilogy(hfields[i], swapsh[i], '.-', color = 'orange')
        plt.xlabel('Magnetic field')
        plt.ylabel('Ratio of swaps')
        plt.title('Ratio of swaps as a function of the magnetic field')
        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/NumberSwapsField_L={0}_various-nsms.png'.format(L[i]))


# In[ ]:


def BasicPlotsE(L, n, tidmin, tidmax, temperatures_plots, hfields_plots, foldername,
                results_foldername, filenamelist, t_h_MeanE, t_h_MeanEsq, t_h_varMeanE,
                t_h_varMeanEsq, C, ErrC, J1, J2, J3, J4, S0 = np.log(2), **kwargs):
    
    ploth = kwargs.get('ploth', False)
    
    t_h_MeanE = np.array(t_h_MeanE)
    t_h_MeanEsq =  np.array(t_h_MeanEsq)
    t_h_varMeanE =  np.array(t_h_varMeanE)
    t_h_varMeanEsq =  np.array(t_h_varMeanEsq)
    C = np.array(C)
    ErrC = np.array(ErrC)
    
    
    # Mean E
    margin = [0.18, 0.2, 0.02, 0.02]
    for i in range(n):
        if ploth:
            mt = len(temperatures_plots[i])
            plt.figure(figsize=(12, 8),dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for tid, t in enumerate(temperatures_plots[i]):
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i],
                                 t_h_MeanE[i][tid, :],'.-',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i],
                                 (t_h_MeanE[i][tid,:]
                                  - np.sqrt(t_h_varMeanE[i][tid,:])),
                                 (t_h_MeanE[i][tid,:]
                                  + np.sqrt(t_h_varMeanE[i][tid,:])),\
                                 alpha=0.4, color = col)
            plt.xlabel(r'Magnetic field $h$')
            plt.ylabel(r'$E$')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.title('Filename: '+filenamelist[i])
            plt.savefig('./' + foldername + 'Plots' + results_foldername
                        + '/h_E.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername
                        + '/h_E.pgf')
        else:
            mh = len(hfields_plots[i])
            plt.figure(figsize=(12, 8),dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                                 t_h_MeanE[i][tidmin:tidmax[i]][:,hid],'.-',\
                                  label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                                 (t_h_MeanE[i][tidmin:tidmax[i]][:,hid]
                                  - np.sqrt(t_h_varMeanE[i][tidmin:tidmax[i]][:,hid])),
                                 (t_h_MeanE[i][tidmin:tidmax[i]][:,hid]
                                  + np.sqrt(t_h_varMeanE[i][tidmin:tidmax[i]][:,hid])),\
                                 alpha=0.4, color = col)
            plt.xlabel(r'Temperature $T$')
            plt.ylabel(r'$E$')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.title('Filename: '+filenamelist[i])
            plt.savefig('./' + foldername + 'Plots' + results_foldername
                        + '/Mean energy per site_various-nsms.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername
                        + '/Mean energy per site_various-nsms.pgf')

    if not ploth:
        #Heat capacity
        margin = [0.18, 0.2, 0.02, 0.02]

        for i in range(n):
            mh = len(hfields_plots[i])
            plt.figure(figsize=(12, 8), dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                             C[i][tidmin:tidmax[i]][:,hid], '.-',\
                             label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                                 C[i][tidmin:tidmax[i]][:,hid]
                                 - ErrC[i][tidmin:tidmax[i]][:,hid],
                                 C[i][tidmin:tidmax[i]][:,hid]
                                 + ErrC[i][tidmin:tidmax[i]][:,hid],\
                                 alpha = 0.5, color = col)
                #print('Error on the heat capacity for file ', filenamelist[i])
                #print(ErrC[i])
            plt.xlabel(r'Temperature $T$ ')
            plt.ylabel(r'Heat capacity $C$ ')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.title('Filename: '+filenamelist[i])
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityErrors_various-nsms.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityErrors.pgf')

        ##Heat capacity / T
        margin = [0.18, 0.2, 0.02, 0.02]

        for i in range(n):
            mh = len(hfields_plots[i])
            plt.figure(figsize=(12, 8), dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                             C[i][tidmin:tidmax[i]][:,hid] / temperatures_plots[i][tidmin:tidmax[i]],
                             '.-', label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                                 (C[i][tidmin:tidmax[i]][:,hid]
                                  - ErrC[i][tidmin:tidmax[i]][:,hid]
                                 )/temperatures_plots[i][tidmin:tidmax[i]],
                                 (C[i][tidmin:tidmax[i]][:,hid]
                                  + ErrC[i][tidmin:tidmax[i]][:,hid]
                                 )/temperatures_plots[i][tidmin:tidmax[i]],\
                                 alpha = 0.5, color = col)
            plt.xlabel(r'Temperature $T$ ')
            plt.ylabel(r'$\frac{c}{k_B T}$')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.title('Filename: '+filenamelist[i])
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityT_various-nsms.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityT.pgf')

        # Residual entropy
        RS = kwargs.get('RS', False)
        if RS:
            S = [[] for i in range(n)]
            DeltaS = [[[0 for hid in range(len(hfields_plots[i]))]
                       for tid in range(tidmax[i]-tidmin)] for i in range(n)]

            for i in range(n):
                Carray = np.array(C[i][tidmin:tidmax[i]])
                CoverT = np.copy(Carray)
                for tid in range(tidmin, tidmax[i]):
                    CoverT[tid,:]= Carray[tid,:]/temperatures_plots[i][tid]

                #going through the temperatures in decreasing order
                for tid in range(tidmax[i]-tidmin-2, -1, -1):
                    for hid, h in enumerate(hfields_plots[i]):
                        DeltaS[i][tid][hid] =                        DeltaS[i][tid+1][hid] + np.trapz(CoverT[tid:tid+2, hid],
                                   temperatures_plots[i][tid+tidmin:tid+2+tidmin])

                DeltaS[i] = np.array(DeltaS[i])
                for tid in range(0, tidmax[i]-tidmin):    
                    S[i].append(S0 - DeltaS[i][tid])

                S[i] = np.array(S[i])

            for i in range(n):
                plt.figure(figsize=(12, 8), dpi=300)
                plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
                for hid, h in enumerate(hfields_plots[i]):
                    col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh] 
                    plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , S[i][:,hid],
                                 '.-', label = r'$h$ = {0}'.format(h), color = col)
                    plt.xlabel(r'Temperature $T$ ')
                plt.ylabel(r'$S$')
                plt.legend(loc= 'best', framealpha=0.5)
                plt.title('Filename: '+filenamelist[i])
                plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/EntropyT.png')
                plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/EntropyT.pgf')

        # Ground-state energy
        gs = kwargs.get('gs', False)
        if gs: 
            r1 = [0, 0.5]
            E1 = [-2/3, -1/6]
            r2 = [0.5, 1]
            E2 = [-1/6, -1/3]
            r3 = [1, 4]
            E3 = [-1/3, -4+2/3]
            margin = [0.18, 0.2, 0.02, 0.02]
            plt.figure(figsize=(9, 5))
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            plt.plot(r1, E1, color = 'orange', label = r'$E - E_{NN}$ = $-\frac{2}{3}$ $J_2$ + $J_3$')
            plt.plot(r3, E3, color = 'red', label = r'$E - E_{NN} = \frac{2}{3}$ $J_2$ - $J_3$')
            plt.plot(r2, E2, '--', color = 'purple', label = r'$E- E_{NN} = -\frac{1}{3}J_3$')
            ratios = list()
            E = list()
            correction = list()
            for i in range(n):
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
            plt.plot(ratios, E, '.-', label = r'Energy at $T = 0.05$ (N)')
            plt.fill_between(ratios , E - correction, E + correction, alpha = 0.5, color = 'lightblue')
            #plt.plot(ratios, [0 for r in ratios], '.')
            plt.xlabel(r'$\frac{J_3}{J_2}$', size = 22)
            plt.ylabel(r'$\frac{E - E_{NN}}{J_2}$', size = 22)
            #plt.legend()
            plt.savefig('./' + foldername + 'Plots' + results_foldername + '/E(ratio)_various-nsms.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername + '/E(ratio).pgf')


# In[ ]:


def BasicPlotsM(L, n, tidmin, tidmax, temperatures_plots, hfields_plots, foldername,
                results_foldername, filenamelist, t_h_MeanM, t_h_MeanMsq, 
                t_h_varMeanM, t_h_varMeanMsq, Chi, ErrChi, J1, J2, J3, J4, **kwargs):
    
    ploth = kwargs.get('ploth', False)
    
    ## Magnetisation
    t_h_MeanM = np.array(t_h_MeanM)
    t_h_MeanMsq =  np.array(t_h_MeanMsq)
    t_h_varMeanM =  np.array(t_h_varMeanM)
    t_h_varMeanMsq =  np.array(t_h_varMeanMsq)
    Chi = np.array(Chi)
    ErrChi = np.array(ErrChi)
    #Magnetisation:
    margin = [0.18, 0.2, 0.02, 0.02]
    for i in range(n):
        if ploth:
            mt = len(temperatures_plots[i])
            plt.figure(figsize=(12, 8),dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for tid, t in enumerate(temperatures_plots[i]):
                col = [0 + tid/mt, (1 - tid/mt)**2, 1 - tid/mt]
                plt.plot(hfields_plots[i],
                                 t_h_MeanM[i][tid, :],'.-',\
                                  label = r'$T$ = {0}'.format(t), color = col)
                plt.fill_between(hfields_plots[i],
                                 (t_h_MeanM[i][tid,:]
                                  - np.sqrt(t_h_varMeanM[i][tid,:])),
                                 (t_h_MeanM[i][tid,:]
                                  + np.sqrt(t_h_varMeanM[i][tid,:])),\
                                 alpha=0.4, color = col)
            plt.xlabel(r'Magnetic field $h$')
            plt.ylabel(r'Magnetisation per site $m$')
            plt.legend(loc= 'best', framealpha=0.5)
            plt.title('Filename: '+filenamelist[i])
            plt.savefig('./' + foldername + 'Plots' + results_foldername
                        + '/h_M.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername
                        + '/h_M.pgf')
        else:
            mh = len(hfields_plots[i])
            plt.figure(figsize=(12, 8), dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]], t_h_MeanM[i][tidmin:tidmax[i]][:,hid], 
                             '.-',label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                                 (t_h_MeanM[i][tidmin:tidmax[i]][:,hid]
                                  - np.sqrt(t_h_varMeanM[i][tidmin:tidmax[i]][:,hid])),
                                 (t_h_MeanM[i][tidmin:tidmax[i]][:,hid]
                                  + np.sqrt(t_h_varMeanM[i][tidmin:tidmax[i]][:,hid])),\
                                 alpha = 0.5, color = col)
            plt.xlabel(r'Temperature $T$ ')
            plt.ylabel('Magnetisation per site')
            plt.title('Filename: '+filenamelist[i])
            plt.legend(loc= 'best', framealpha=0.5)
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/M_various-nsms.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/M.pgf')
    
    if not ploth:
        #Susceptibility
        for i in range(n):
            mh = len(hfields_plots[i])
            plt.figure(figsize=(12, 8), dpi=300)
            plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
            for hid, h in enumerate(hfields_plots[i]):
                col = [0 + hid/mh, (1 - hid/mh)**2, 1 - hid/mh]
                plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]],
                             Chi[i][tidmin:tidmax[i]][:,hid], '.-',\
                             label = r'$h$ = {0}'.format(h), color = col)
                plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],
                                 Chi[i][tidmin:tidmax[i]][:,hid]
                                 - ErrChi[i][tidmin:tidmax[i]][:,hid], 
                                 Chi[i][tidmin:tidmax[i]][:,hid]
                                 + ErrChi[i][tidmin:tidmax[i]][:,hid],
                                 alpha = 0.5, color = col)
            plt.xlabel(r'Temperature $T$ ')
            plt.ylabel('Susceptibility')
            plt.title('Filename: '+filenamelist[i])
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/Susceptibility_various-nsms.png')
            plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/Susceptibility.pgf')


# In[ ]:


def Compute2DCorrelations(rid, n, t_h_MeanCorr,
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
                corr[i][tid] = np.array(t_h_MeanCorr[i])[:,tid,rid]
                errcorr[i][tid] = np.sqrt(np.array(t_h_errCorrEstim[i])[:,tid,rid])
                maxerr[i][tid] = np.amax(np.abs(np.array(t_h_MeanSi[i])[:,tid,rid]))**2

    return corr, errcorr, maxerr


# In[ ]:


def BasicPlotsCorrelations2D(foldername, results_foldername, rid,
                             n, L, corr, errcorr, t_h_MeanSi,
                             hfields_plots, temperatures_plots,\
                             ploth = False):
    if not ploth:
        matplotlib.rcParams.update({'font.size': 6})
        for i in range(n):
            a =1
            for hid, h in enumerate(hfields_plots[i]):
                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][hid][0], L[i], a)
                plt.title('L = {0}; h = {1}'.format(L[i], h))
                plt.clim(-1,1)
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations0_L{0}_h={1}.png'.format(L[i], h))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations0_L{0}_h={1}}.pgf'.format(L[i],h))
                plt.show()


                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][hid][1], L[i], a)
                plt.title('L = {0}; h = {1}'.format(L[i], h))
                plt.clim(-1,1)
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations1_L{0}_h={1}.png'.format(L[i],h))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername +                            '/Correlations1_L{0}_h={1}.pgf'.format(L[i],h))
                plt.show()

                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][hid][2], L[i], a)
                plt.title('L = {0}; h = {1}'.format(L[i], h))
                plt.clim(-1,1)
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations_L{0}_h={1}.png'.format(L[i],h))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations_L{0}_h={1}.pgf'.format(L[i],h))
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
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Spinaverage_L{0}_h={1}.png'.format(L[i],h))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Spinaverage_L{0}_h={1}.pgf'.format(L[i],h))
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
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations0_L{0}_t={1}.png'.format(L[i],t))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations0_L{0}_t={1}.pgf'.format(L[i],t))
                plt.show()


                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][tid][1], L[i], a)
                plt.title('L = {0}; t = {1}'.format(L[i], t))
                plt.clim(-1,1)
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations1_L{0}_t={1}.png'.format(L[i],t))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations1_L{0}_t={1}.pgf'.format(L[i],t))
                plt.show()

                plt.figure(dpi=300)
                kdraw.plot_function_kag(corr[i][tid][2], L[i], a)
                plt.title('L = {0}; t = {1}'.format(L[i], t))
                plt.clim(-1,1)
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations2_L{0}_t={1}.png'.format(L[i],t))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Correlations2_L{0}_t={1}.pgf'.format(L[i],t))
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
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Spinaverage_L{0}_t={1}.png'.format(L[i],t))
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/Spinaverage_L{0}_t={1}.pgf'.format(L[i],t))
                plt.show()
                
        #### PLOTTING ERRORS ON CORRELATIONS IN 2D
        #for i in range(n):
        #    a = 1
        #    for hid, h in enumerate(hfields_plots[i]):
#
#
        #        plt.figure(dpi=150)
        #        kdraw.plot_function_kag(errcorr[i][hid][0], L[i], a)
        #        #plt.clim(-1,1)
        #        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/CorrelationsErr0_L{0}_various-nsms.png'.format(L[i]))
        #        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/CorrelationsErr0_L{0}.pgf'.format(L[i]))
        #        plt.title('L = {0}; h = {1}'.format(L[i], h))
        #        plt.show()
#
#
        #        plt.figure(dpi=150)
        #        kdraw.plot_function_kag(errcorr[i][hid][1], L[i], a)
        #        #plt.clim(-1,1)
        #        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/CorrelationsErr1_L{0}_various-nsms.png'.format(L[i]))
        #        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/CorrelationsErr1_L{0}.pgf'.format(L[i]))
        #        plt.title('L = {0}; h = {1}'.format(L[i], h))
        #        plt.show()
#
        #        plt.figure(dpi=150)
        #        kdraw.plot_function_kag(errcorr[i][hid][2], L[i], a)
        #        #plt.clim(-1,1)
        #        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/CorrelationsErr2_L{0}_various-nsms.png'.format(L[i]))
        #        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/CorrelationsErr2_L{0}.pgf'.format(L[i]))
        #        plt.title('L = {0}; h = {1}'.format(L[i], h))
        #        plt.show()


# In[ ]:


def PlotStrctFact(StrctFact, foldername, results_foldername, tid,
                  hid,L, i, hfields_plots, temperatures_plots):
    size = (170/L[i])**2
    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,0,0]),
                                   L[i], 2, s = size)#, vmin = vmin, vmax = vmax)
    plt.title('L = {0}; h = {1}; SF 00'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername + 'Plots' + results_foldername+                '/SF00_h={0}_L={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,1,1]),
                                   L[i], 2, s = size)#, vmin = vmin, vmax = vmax)
    plt.title('L = {0}; h = {1}; SF 11'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername + 'Plots' + results_foldername+                '/SF11_h={0}_L={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,2,2]),
                                   L[i], 2, s = size)#, vmin = vmin, vmax = vmax)
    plt.title('L = {0}; h = {1}; SF 22'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername + 'Plots' + results_foldername+                '/SF22_h={0}_L={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,0,1]
                                           +StrctFact[:,1,0])/2,
                                   L[i], 2, s = size)#, vmin = vmin, vmax = vmax)
    plt.title('L = {0}; h = {1}; SF 01 + 10'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername + 'Plots' + results_foldername+                '/SF01_h={0}_L={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,0,2]
                                           +StrctFact[:,2,0])/2,
                                   L[i], 2, s = size)#, vmin = vmin, vmax = vmax)
    plt.title('L = {0}; h = {1}; SF 02 + 20'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername + 'Plots' + results_foldername+                '/SF02_h={0}_L={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    kdraw.plot_function_reciprocal(np.real(StrctFact[:,1,2]
                                           +StrctFact[:,2,1])/2,
                                   L[i], 2, s = size)#, vmin = vmin, vmax = vmax)
    plt.title('L = {0}; h = {1}; SF 12 + 21'.format(L[i], hfields_plots[i][hid]))
    plt.savefig('./' + foldername + 'Plots' + results_foldername+                '/SF12_h={0}_L={1}_t={2}.png'.format(L[i],
                                                     hfields_plots[i][hid],
                                                     temperatures_plots[i][tid]))


# In[ ]:


def dist_corr(L, findex, corr, errcorr,distmax):
    distances, distances_spins, NNList, s_pos, srefs = kf.NearestNeighboursLists(L, distmax)
    
    C = [[0 for i in range(len(NNList[0]))] for j in range(len(srefs))]
    ErrC = [[0 for i in range(len(NNList[0]))] for j in range(len(srefs))]
    for j in range(len(srefs)):
        for i in range(len(NNList[0])):
            Corrji = 0
            ErrCorrji = 0
            count = 0
            for pair in NNList[j][i]:
                if srefs[j] == pair[0]:
                    count += 1
                    Corrji += corr[findex][j][pair[1]]
                    ErrCorrji += errcorr[findex][j][pair[1]]
            if count == 0:
                print("NNList[", j, "][", i, "] = ", NNList[j][i])
            Corrji = Corrji/count
            ErrCorrji = ErrCorrji/count

            
            C[j][i] = Corrji
            ErrC[j][i]= ErrCorrji
        C[j] = np.array(C[j])
        ErrC[j] = np.array(ErrC[j])

    C = np.array(sum(C))/3
    ErrC = np.array(sum(ErrC))/3
    return distances, C, ErrC
        


# In[ ]:


def PlotFirstCorrelations(n, L, foldername, results_foldername,hfields_plots, temperatures_plots,
                         t_h_MeanCorr, t_h_errCorrEstim, distmax = 3.5, ploth = False):

    distmax = min(3.5, distmax)
    nlistnames = ['1', '2', '3', '3star', '4', '5', '6', '6star']

    if not ploth:
        for i in range(n):
            for hid, h in enumerate(hfields_plots[i]):
                fig, ax = plt.subplots(dpi=200, figsize = (9,9))
                ax.set_xscale("log")
                plt.title('First few neighbours correlations,                h = {0}'.format(h))
                fmts = ['.','x','v','-','^','o','*','s']
                length = len(temperatures_plots[i])
                fmt = fmts[i]
                for t in range(1,length):

                    corr = [np.array(t_h_MeanCorr[i])[:,t,hid,:]]
                    errcorr =                    [np.sqrt(np.array(t_h_errCorrEstim[i])[:,t,hid])]
                    (resr, rescorr, reserrcorr) =                    dist_corr(L[i],0 ,corr, errcorr, distmax)
                    
                    if t == 1:
                            print(rescorr)
                    
                    plt.gca().set_prop_cycle(None)
                    alpha = 0.5
                    for nei in range(0,len(rescorr)):
                        if t == 1:
                            plt.errorbar(temperatures_plots[i][t],
                                         rescorr[nei],
                                         reserrcorr[nei],\
                                         fmt = fmt,\
                                         label =\
                                         'Neighbour {0}'.format(nlistnames[nei]),\
                                         alpha = alpha)
                        else:
                            plt.errorbar(temperatures_plots[i][t],
                                         rescorr[nei],
                                         reserrcorr[nei],\
                                         fmt = fmt,\
                                         alpha = alpha)

                plt.xlabel(r'$T/J_1$')
                plt.ylabel(r'$<\sigma_i \sigma_j> - <\sigma_i> <\sigma_j> $')
                plt.legend(loc = 'best')
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/FewCorrelations_L={0}_h={1}.png'.format(L[i],h))
    else:
        for i in range(n):
            for tid, t in enumerate(temperatures_plots[i]):
                fig, ax = plt.subplots(dpi=200, figsize = (9,9))
                plt.title('First few neighbours correlations,                t = {0}'.format(t))
                fmts = ['.','x','v','-','^','o','*','s']
                length = len(hfields_plots[i])
                fmt = fmts[i]
                for hid in range(1,length):

                    corr = [np.array(t_h_MeanCorr[i])[:,tid,hid,:]]
                    errcorr =                    [np.sqrt(np.array(t_h_errCorrEstim[i])[:,tid,hid])]
                    (resr, rescorr, reserrcorr) =                    dist_corr(L[i],0 ,corr, errcorr, distmax)
                    
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
                                         'Neighbour {0}'.format(nlistnames[nei]),\
                                         alpha = alpha)
                        else:
                            plt.errorbar(hfields_plots[i][hid],
                                         rescorr[nei],
                                         reserrcorr[nei],\
                                         fmt = fmt,\
                                         alpha = alpha)

                plt.xlabel(r'$h/J_1$')
                plt.ylabel(r'$<\sigma_i \sigma_j> - <\sigma_i> <\sigma_j> $')
                plt.legend(loc = 'best')
                plt.savefig('./' + foldername + 'Plots' +                            results_foldername+                            '/FewCorrelations_L={0}_t={1}.png'.format(L[i],t))
 

