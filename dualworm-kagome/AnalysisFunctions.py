
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pickle
import KagomeFunctions as kf # "library" allowing to work on Kagome
import DualwormFunctions as dw


# In[2]:


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
    
    listfunctions = [[] for _ in range(n)]
    
    sref = [[] for _ in range(n)]
    
    for nf, filename in enumerate(filenamelist):
        [L[nf], numsites[nf], J1[nf], J2[nf], J3[nf], J3st[nf], J4[nf], nb[nf], 
         num_in_bin[nf], temperatures[nf], nt[nf], stat_temps[nf], temperatures_plots[nf],
         listfunctions[nf], sref[nf]] = LoadParametersFromFile(foldername, filename)
    
    return L, numsites, J1, J2, J3, J3st, J4, nb, num_in_bin, temperatures, nt,             stat_temps, temperatures_plots, listfunctions, sref


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
    
    listfunctions = backup.results.namefunctions
    
        
    #reference spins
    s0 = ijl_s[L, L, 0]
    s1 = ijl_s[L, L, 1]
    s2 = ijl_s[L, L, 2]
    
    sref = [s0, s1, s2]
    f.close()
    
    return L, numsites, J1, J2, J3, J3st, J4, nb, num_in_bin, temperatures, nt,             stat_temps, temperatures_plots, listfunctions, sref


# In[ ]:


def ExtractStatistics(idfunc, meanstat, nb, stat_temps, sq = 0, **kwargs):
    '''
        This function gets the statistics from a file and
        computes the expectation values and variances of 
        the operator corresponding to idfunc
        
        sq = 0 -> not square stats
        sq = 1 -> square stats
    '''
    stattuple = meanstat[idfunc];
    t_meanfunc = np.array(stattuple[sq]).sum(1)/nb
    
    binning = kwargs.get('binning', False)

    t_varmeanfunc = [0 for t in stat_temps]
    for resid, t in enumerate(stat_temps):
        for b in range(nb):
            t_varmeanfunc[resid] += ((stattuple[sq][resid][b] - t_meanfunc[resid]) ** 2)/(nb * (nb - 1))
    if binning:
        print('binning!')
        Binning(t_meanfunc,t_varmeanfunc, stattuple[sq], nb,
                                stat_temps, **kwargs)
        
    return t_meanfunc, t_varmeanfunc


# In[ ]:


def Binning(t_mean, t_varmean, stattuple, nb, stat_temps, **kwargs):
    '''
        This function implements a binning analysis
    '''
    ### NAIVE IMPLEMENTATION
    nblist = []
    nbb = nb
    while nbb >= 15:
        nblist.append(nbb)
        nbb = nbb//2
        
        
    t_vars = [[] for resid in range(len(stat_temps))]
    for resid, t in enumerate(stat_temps):
        var = []
        for l,nbb in enumerate(nblist):
            avg = np.array(stattuple[resid][0:(2**l)]).sum(0)/(2**l)
            varl=((avg - t_mean[resid])**2)/(nbb*(nbb-1))
            for b in range(1,nbb):
                avg = np.array(stattuple[resid][(2**l)*b:(2**l)*(b+1)]).sum(0)/(2**l)
                varl+=((avg - t_mean[resid])**2)/(nbb*(nbb-1))
            if len(varl.shape) == 0:
                var.append(varl)
            else:
                var.append(np.max(varl))
            if resid == 0:
                print(nbb, " --- ", var[l])
           
        t_vars[resid] = var
            
    plzplot = kwargs.get('plzplot', False)
    plotmin = kwargs.get('plotmin', 0)
    plotmax = kwargs.get('plotmax', 10)
    if plzplot:
        print('plotting!')
        plt.figure(figsize=(18, 12),dpi=300)
        minplt = max(0, plotmin)
        maxplt = min(plotmax, len(stat_temps))
        for resid, t in enumerate(stat_temps[minplt:maxplt]):
            plt.plot(range(len(t_vars[resid])), t_vars[resid], '.-', label = 't = {0}'.format(t))
        plt.legend()
        plt.show()
    t_varmean = [max(var) for var in t_vars] 


# In[ ]:


def LoadSwaps(foldername, filenamelist, nb, num_in_bin):
    n = len(filenamelist)
    swapsth = [[] for _ in range(n)]
    swaps = [[] for _ in range(n)]

    for nf, filename in enumerate(filenamelist):
        [swapsth[nf], swaps[nf]] = LoadSwapsFromFile(foldername, filename, nb[nf], num_in_bin[nf])
    
    return swapsth, swaps


# In[ ]:


def LoadSwapsFromFile(foldername, filename, nb, num_in_bin):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f)
    
    nsms = nb*num_in_bin
    swapsth = backup.results.swapsth
    swaps = np.array(backup.results.swaps)/(2*nsms)
    
    f.close()
    return swapsth, swaps


# In[ ]:


def LoadEnergy(foldername, filenamelist, numsites, nb, stat_temps, temperatures, listfunctions, **kwargs):
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
            [t_MeanE[nf], t_MeanEsq[nf], t_varMeanE[nf], t_varMeanEsq[nf], C[nf], ErrC[nf]] =                 LoadEnergyFromFile(foldername, filename, numsites[nf], nb[nf], stat_temps[nf], temperatures[nf], idfunc, **kwargs)
        else:
            [t_MeanE[nf], t_MeanEsq[nf], t_varMeanE[nf], t_varMeanEsq[nf], C[nf], ErrC[nf]] = [[],[],[],[],[],[]]
        
    return t_MeanE, t_MeanEsq, t_varMeanE, t_varMeanEsq, C, ErrC


# In[ ]:


def LoadEnergyFromFile(foldername, filename, numsites, nb, stat_temps, temperatures, idfunc, **kwargs):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) 
    
    meanstat = backup.results.meanstat
    #t_meanfunc = backup.results.t_meanfunc
    #t_varmeanfunc = backup.results.t_varmeanfunc
    
    t_MeanE, t_varMeanE = ExtractStatistics(idfunc, meanstat, nb, stat_temps, **kwargs)
    t_MeanEsq, t_varMeanEsq = ExtractStatistics(idfunc, meanstat, nb, stat_temps, sq = 1, **kwargs)
    #t_MeanE = t_meanfunc[idfunc][0].tolist()
    #t_MeanEsq = t_meanfunc[idfunc][1].tolist()
    #t_varMeanE = t_varmeanfunc[idfunc][0]
    #t_varMeanEsq = t_varmeanfunc[idfunc][1]
    
    C = []
    ErrC = []
    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        C.append(numsites * (t_MeanEsq[resid] - t_MeanE[resid] ** 2) / T ** 2)

    # to compute the error on C, we need to compute sqrt(<C^2> - <C>^2) where "<>" stands for the average over all the bins
    # i.e. <C> = 1/nb * sum_b C_b where C_b is the value of C over the bin b
    # Note that C_b = N/T^2 * (<E^2>_b - <E>_b ^2) where <>_b stands for the average over bin b

    tb_E = meanstat[idfunc][0]
    tb_Esq = meanstat[idfunc][1]

    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        Mean_VarE = 0
        Mean_VarE_Sq = 0
        for b in range(nb):
                Mean_VarE += (tb_Esq[resid][b] - tb_E[resid][b] ** 2)/nb
                Mean_VarE_Sq += ((tb_Esq[resid][b] - tb_E[resid][b] ** 2) ** 2)/nb
        if (Mean_VarE_Sq - Mean_VarE ** 2 >= 0) :
            ErrC.append(numsites / (T ** 2) * np.sqrt(Mean_VarE_Sq - Mean_VarE ** 2))
        else:
            assert(Mean_VarE_Sq - Mean_VarE ** 2 >= -1e-15)
            ErrC.append(0)
            
    f.close()
    
    return t_MeanE, t_MeanEsq, t_varMeanE, t_varMeanEsq, C, ErrC


# In[ ]:


def LoadMagnetisation(foldername, filenamelist, numsites, nb, stat_temps, temperatures, listfunctions, **kwargs):
    n = len(filenamelist)
    
    t_MeanM = [[] for _ in range(n)]
    t_MeanMsq = [[] for _ in range(n)]
    t_varMeanM = [[] for _ in range(n)]
    t_varMeanMsq = [[] for _ in range(n)]
    Chi = [[] for _ in range(n)]
    ErrChi = [[] for _ in range(n)]

    for nf, filename in enumerate(filenamelist):
        if 'Magnetisation' in listfunctions[nf]:
            idfunc = listfunctions[nf].index('Magnetisation')
            [t_MeanM[nf], t_MeanMsq[nf], t_varMeanM[nf], t_varMeanMsq[nf], Chi[nf], ErrChi[nf]] =                 LoadMagnetisationFromFile(foldername, filename, numsites[nf], nb[nf], stat_temps[nf], temperatures[nf], idfunc, **kwargs)
        else:
            [t_MeanM[nf], t_MeanMsq[nf], t_varMeanM[nf], t_varMeanMsq[nf], Chi[nf], ErrChi[nf]] = [[],[],[],[],[],[]]
        
    return t_MeanM, t_MeanMsq, t_varMeanM, t_varMeanMsq, Chi, ErrChi


# In[ ]:


def LoadMagnetisationFromFile(foldername, filename, numsites, nb, stat_temps, temperatures, idfunc,  **kwargs):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) 
    
    meanstat = backup.results.meanstat
    
    t_MeanM, t_varMeanM = ExtractStatistics(idfunc, meanstat, nb, stat_temps, **kwargs)
    t_MeanMsq, t_varMeanMsq = ExtractStatistics(idfunc, meanstat, nb, stat_temps, sq = 1, **kwargs)
    Chi = []
    ErrChi = []
    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        Chi.append(numsites * (t_MeanMsq[resid] - t_MeanM[resid] ** 2) / T)

    tb_M = meanstat[idfunc][0]
    tb_Msq = meanstat[idfunc][1]

    for resid, t in enumerate(stat_temps):
        T = temperatures[t]
        Mean_VarM = 0
        Mean_VarM_Sq = 0
        for b in range(nb):
                Mean_VarM += (tb_Msq[resid][b] - tb_M[resid][b] ** 2)/nb
                Mean_VarM_Sq += ((tb_Msq[resid][b] - tb_M[resid][b] ** 2) ** 2)/nb
        if Mean_VarM_Sq - Mean_VarM ** 2 >= 0 :
            ErrChi.append(numsites / T * np.sqrt(Mean_VarM_Sq - Mean_VarM ** 2))  
        else:
            assert(Mean_VarM_Sq - Mean_VarM ** 2 >= -1e-15)
            ErrChi.append(0)
            
    f.close()
    
    return t_MeanM, t_MeanMsq, t_varMeanM, t_varMeanMsq, Chi, ErrChi


# In[ ]:


def LoadCentralCorrelations(foldername, filenamelist, listfunctions, sref, stat_temps, nb, **kwargs):
    n = len(filenamelist)
    
    ## "Correlations" <sisj>
    t_MeanSs = [[] for _ in range(n)]
    t_varMeanSs = [[] for _ in range(n)]
    
    ## Local spin average
    t_MeanSi = [[] for _ in range(n)]
    t_varMeanSi = [[] for _ in range(n)]
    
    ## Correlations
    t_MeanCorr = [[] for _ in range(n)]
    t_errCorrEstim = [[] for _ in range(n)]
    for nf, filename in enumerate(filenamelist):
        if ('Central_Correlations' in listfunctions[nf] and 'Si' in listfunctions[nf]): # This will be improved when we will work with a more general way of handling the correlations
            idfunc = listfunctions[nf].index('Central_Correlations')
            idfuncsi = listfunctions[nf].index('Si')

            [t_MeanSs[nf], t_varMeanSs[nf], t_MeanSi[nf], t_varMeanSi[nf], t_MeanCorr[nf], 
             t_errCorrEstim[nf]] = LoadCorrelationsFromFile(foldername, filename, idfunc, idfuncsi, sref[nf], stat_temps[nf], nb[nf], **kwargs)
        else:
            [t_MeanSs[nf], t_varMeanSs[nf], t_MeanSi[nf], t_varMeanSi[nf], t_MeanCorr[nf], 
             t_errCorrEstim[nf]] = [[],[],[],[],[]]
    return t_MeanSs, t_varMeanSs, t_MeanSi, t_varMeanSi, t_MeanCorr, t_errCorrEstim


# In[ ]:


def LoadCorrelationsFromFile(foldername, filename, idfunc, idfuncsi, sref, stat_temps, nb, **kwargs):
    f = open('./' + foldername + filename +'.pkl', 'rb')
    backup = pickle.load(f) 
    
    meanstat = backup.results.meanstat
    #t_meanfunc = backup.results.t_meanfunc
    #t_varmeanfunc = backup.results.t_varmeanfunc
    
    # Averages and corresponding variances
    #t_MeanSi = t_meanfunc[idfuncsi][0] # <si>
    #t_varMeanSi = t_varmeanfunc[idfuncsi][0] # var(<si>)
    t_MeanSi, t_varMeanSi = ExtractStatistics(idfuncsi, meanstat, nb, stat_temps, **kwargs)
    
    #t_MeanSs = t_meanfunc[idfunc][0]
    #t_varMeanSs = t_varmeanfunc[idfunc][0]
    t_MeanSs, t_varMeanSs = ExtractStatistics(idfunc, meanstat, nb, stat_temps, **kwargs)
    
    
    t_MeanCorr = []
    for i in range(len(sref)):
        column = t_MeanSi[:, sref[i]]
        column = column[:,np.newaxis]
        t_MeanCorr.append(t_MeanSs[:,i,:] - t_MeanSi*column) #<si sj> - <si> <sj> for j in lattice. /!\ this is ELEMENTWISE
        
    # Estimating the error on <si sj> - <si><sj>
    t_errCorrEstim = CorrelErrorEstimator(meanstat, idfunc, idfuncsi, sref, stat_temps, nb)   
 
    f.close()
    return t_MeanSs, t_varMeanSs, t_MeanSi, t_varMeanSi, t_MeanCorr, t_errCorrEstim


# In[ ]:


def CorrelErrorEstimator(meanstat, idfunc, idfuncsi, sref, stat_temps, nb):
    t_b_sisj = np.array(meanstat[idfunc][0]) # <s0sj>_b (t)
    (ntm, nb, nrefs, nsites) = t_b_sisj.shape #getting the system size
    
    t_b_sj = np.array(meanstat[idfuncsi][0]) # <si>_b (t)
    t_b_s0 = t_b_sj[:,:,sref]
    
    t_b_gamma = [[] for _  in range(nrefs)]
    t_gamma = [np.zeros((ntm, nsites)) for _ in range(nrefs)]
    for i in range(nrefs):
        t_b_s0i = t_b_s0[:,:,i]
        t_b_s0i = t_b_s0i[:,:,np.newaxis]
        
        for b in range(nb):
            t_b_gamma[i].append(t_b_sisj[:,b,i,:] - t_b_s0i[:,b]*t_b_sj[:,b,:])
            t_gamma[i] += t_b_gamma[i][b]
        t_gamma[i] = t_gamma[i]/nb
    
    
    t_vargamma = [np.zeros((ntm, nsites)) for _ in range(nrefs)]
    for i in range(nrefs):
        for b in range(nb):
            t_vargamma[i] += np.power((t_b_gamma[i][b] - t_gamma[i]),2)
        t_vargamma[i] = t_vargamma[i]/(nb*(nb-1))

    return t_vargamma


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
    
    meanstat = backup.results.meanstat
    
    t_MeanSi, t_varMeanSi = ExtractStatistics(idfuncsi, meanstat, nb, stat_temps, **kwargs)
    
    f.close()
    
    return t_MeanSi, t_varMeanSi
    


# In[ ]:


def SwapsAnalysis(L, n, tidmin, tidmax, temperatures, foldername, results_foldername, swaps):
    for i in range(n):
        plt.figure()
        plt.loglog(temperatures[i][tidmin:tidmax[i]], swaps[i][tidmin:tidmax[i]], '.-', color = 'green')
        plt.xlabel('Temperature')
        plt.ylabel('Number of swaps')
        plt.title('Number of swaps as a function of the temperature')
        plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/NumberSwaps_L={0}_various-nsms.png'.format(L[i]))


# In[ ]:


def BasicPlotsE(L, n, tidmin, tidmax, temperatures_plots, foldername, results_foldername, filenamelist, t_MeanE, t_MeanEsq, t_varMeanE, t_varMeanEsq, C, ErrC, J1, J2, J3, J4):
    t_MeanE = np.array(t_MeanE)
    t_MeanEsq =  np.array(t_MeanEsq)
    t_varMeanE =  np.array(t_varMeanE)
    t_varMeanEsq =  np.array(t_varMeanEsq)
    C = np.array(C)
    ErrC = np.array(ErrC)
    
    # Mean E
    margin = [0.18, 0.2, 0.02, 0.02]
    plt.figure(figsize=(18, 12),dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        col = [0 + i/n, (1 - i/n)**2, 1 - i/n]
        if J2[i] != 0:
            ratio = J3[i]/J2[i]
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , t_MeanE[i][tidmin:tidmax[i]], '.-', label = r'$J_3 / J_2$ = {:f}'.format(ratio), color = col)
        else:
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , t_MeanE[i][tidmin:tidmax[i]], '.-', label = r'$J_3 / J_2$ = $\infty$', color = col)
        plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]], (t_MeanE[i][tidmin:tidmax[i]] - np.sqrt(t_varMeanE[i][tidmin:tidmax[i]])), (t_MeanE[i][tidmin:tidmax[i]] + np.sqrt(t_varMeanE[i][tidmin:tidmax[i]])), alpha=0.4, color = col)
    plt.xlabel(r'Temperature $T$')
    plt.ylabel(r'$E$')
    plt.legend(loc= 'best', framealpha=0.5)
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/Mean energy per site_various-nsms.png')
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/Mean energy per site_various-nsms.png')
    
    #Heat capacity
    margin = [0.18, 0.2, 0.02, 0.02]
    plt.figure(figsize=(18, 12), dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        col = [0 + i/n, (1-i/n) **2, 1 -  i/n]
        if J2[i] != 0:
            ratio = J3[i]/J2[i]
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , C[i][tidmin:tidmax[i]], '.-', label = r'$J_3 / J_2$ = {:f}'.format(ratio), color = col)
        else:
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , C[i][tidmin:tidmax[i]], '.-', label = r'$J_3 / J_2$ = $\infty$', color = col)
        plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]], C[i][tidmin:tidmax[i]] - ErrC[i][tidmin:tidmax[i]], C[i][tidmin:tidmax[i]] + ErrC[i][tidmin:tidmax[i]], alpha = 0.5, color = col)
        #print('Error on the heat capacity for file ', filenamelist[i])
        #print(ErrC[i])
    plt.xlabel(r'Temperature $T$ ')
    plt.ylabel(r'Heat capacity $C$ ')
    plt.legend(loc= 'best', framealpha=0.5)
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityErrors_various-nsms.png')
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityErrors.pgf')

    #Heat capacity
    margin = [0.18, 0.2, 0.02, 0.02]
    plt.figure(figsize=(18, 12), dpi=300)
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        col = [0 + i/n, (1- i/n)**2, 1 -  i/n]
        if J2[i] != 0:
            ratio = J3[i]/J2[i]    
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , C[i][tidmin:tidmax[i]] / temperatures_plots[i][tidmin:tidmax[i]], '.-', label = r'$J_3 / J_2$ = {:f}'.format(ratio), color = col)
        else:
            plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , C[i][tidmin:tidmax[i]] / temperatures_plots[i][tidmin:tidmax[i]], '.-', label = r'$J_3 / J_2$ = $\infty$', color = col)
        plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]], (C[i][tidmin:tidmax[i]] - ErrC[i][tidmin:tidmax[i]])/temperatures_plots[i][tidmin:tidmax[i]], (C[i][tidmin:tidmax[i]] + ErrC[i][tidmin:tidmax[i]])/temperatures_plots[i][tidmin:tidmax[i]], alpha = 0.5, color = col)
    plt.xlabel(r'Temperature $T$ ')
    plt.ylabel(r'$\frac{c}{k_B T}$')
    plt.legend(loc= 'best', framealpha=0.5)
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityT_various-nsms.png')
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/HeatCapacityT.pgf')
    
    # Ground-state energy
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
            E.append((t_MeanE[i][0] + 2/3 * J1[i])/J2[i])
            correction.append(t_varMeanE[i][0]/J2[i])
            #print(t_MeanE[i][0] + 2/3 * J1[i]+J3[i])
        else:
            print(t_MeanE[i][0] + 2/3 * J1[i]+J3[i])
        
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


def BasicPlotsM(L, n, tidmin, tidmax, temperatures_plots, foldername, results_foldername, filenamelist, t_MeanM, t_MeanMsq, t_varMeanM, t_varMeanMsq, Chi, ErrChi, J1, J2, J3, J4):
    ## Magnetisation
    t_MeanM = np.array(t_MeanM)
    t_MeanMsq =  np.array(t_MeanMsq)
    t_varMeanM =  np.array(t_varMeanM)
    t_varMeanMsq =  np.array(t_varMeanMsq)
    Chi = np.array(Chi)
    ErrChi = np.array(ErrChi)
    #Magnetisation:
    margin = [0.18, 0.2, 0.02, 0.02]
    plt.figure(figsize=(6, 4))
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]], t_MeanM[i][tidmin:tidmax[i]], '.-')
        plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]],(t_MeanM[i][tidmin:tidmax[i]] - np.sqrt(t_varMeanM[i][tidmin:tidmax[i]])), (t_MeanM[i][tidmin:tidmax[i]] + np.sqrt(t_varMeanM[i][tidmin:tidmax[i]])), alpha=0.4)
    plt.xlabel(r'Temperature $T$ ')
    plt.ylabel('Magnetisation per site')
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/M_various-nsms.png')
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/M.pgf')
    
    #Susceptibility
    margin = [0.18, 0.2, 0.02, 0.02]
    plt.figure(figsize=(6, 4))
    plt.axes(margin[:2] + [1-margin[0]-margin[2], 1-margin[1]-margin[3]])
    for i in range(n):
        plt.semilogx(temperatures_plots[i][tidmin:tidmax[i]]  , Chi[i][tidmin:tidmax[i]], '.-')
        plt.fill_between(temperatures_plots[i][tidmin:tidmax[i]], Chi[i][tidmin:tidmax[i]] - ErrChi[i][tidmin:tidmax[i]], Chi[i][tidmin:tidmax[i]] + ErrChi[i][tidmin:tidmax[i]], alpha = 0.5)
    plt.xlabel(r'Temperature $T$ ')
    plt.ylabel('Susceptibility')
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/Susceptibility_various-nsms.png')
    plt.savefig('./' + foldername + 'Plots' + results_foldername+ '/Susceptibility.pgf')


# In[ ]:


def PBCKagomeLatticeNeighboursLists(s0, s1, s2, s_pos, distconds):
    '''
        Given the reference sites s0, s1 and s2, and given the spin index <-> position
        table s_pos and the conditions delimitating the various distances at which
        there are neighbours, returns the lists of neighbours for each spin.
    '''
    nn = len(distconds)

