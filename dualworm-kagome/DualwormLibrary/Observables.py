import numpy as np

def energy(stlen, state, en_state, spinstate, s_ijl, ijl_s, **kwargs):
    return en_state/stlen

def magnetisation(stlen, state, en_state, spinstate, s_ijl, ijl_s,**kwargs):
    M = np.sum(spinstate)
    return M/stlen

def centralcorrelations(stlen, state, en_state, spinstate, s_ijl, ijl_s, srefs = [],**kwargs):
    ref_spin = [spinstate[srefs[0]], spinstate[srefs[1]], spinstate[srefs[2]]]
    central_corr = [ref_spin[0]*spinstate, ref_spin[1]*spinstate, ref_spin[2]*spinstate]
    
    return np.array(central_corr) 

def si(stlen, state, en_state, spinstate, s_ijl, ijl_s,**kwargs):
    return np.array(spinstate, dtype = 'int8')

def firstcorrelations(stlen, state, en_state, spinstate, s_ijl, ijl_s, nnlists = [], **kwargs):
    firstcorr = [sum([spinstate[s1]*spinstate[s2] for (s1,s2) in nnlist])/len(nnlist) for nnlist in nnlists]
    return np.array(firstcorr)

def charges(stlen, state, en_state, spinstate, s_ijl, ijl_s,c2s = [], csign= [], **kwargs):
    cvals = np.array([csign[c]*(spinstate[s1]+spinstate[s2]+spinstate[s3]) for c, (s1,s2,s3) in enumerate(c2s)], dtype = 'int8')
    return cvals

def frustratedTriangles(stlen, state, en_state, spinstate, s_ijl, ijl_s,c2s = [], csign= [], **kwargs):
    numcharges = len(c2s)
    nfr = np.array([1 for c, (s1,s2,s3) in enumerate(c2s) if abs(spinstate[s1]+spinstate[s2]+spinstate[s3]) == 3]).sum()
    return nfr/numcharges

def initstatstables(namefunctions, nb, c2s, nnlists,
                    stat_temps, stat_fields, stlen):
    statstables = [0 for i in range(len(namefunctions))]
    for i, name in enumerate(namefunctions):
        if name == "Energy" or name == "Magnetisation" or name == "FrustratedTriangles":
            statstables[i]=np.zeros((nb, 2, len(stat_temps), len(stat_fields)))
        elif name == "Charges":
            statstables[i] = np.zeros((nb, 2, len(stat_temps),
                                       len(stat_fields), len(c2s)))
        elif name == "Central_Correlations":
            statstables[i] = np.zeros((nb, 2, len(stat_temps),
                                       len(stat_fields), 3, stlen))
        elif name == "Si":
            statstables[i] = np.zeros((nb, 2, len(stat_temps),
                                       len(stat_fields),stlen))    
        elif name == "FirstCorrelations":
            statstables[i] = np.zeros((nb, 2, len(stat_temps),
                                       len(stat_fields), len(nnlists)))
    
    return statstables

def initautocorrel_spins(nit, stat_temps, stat_fields):
    autocorrelation_spins = np.zeros((nit, len(stat_temps), len(stat_fields)))
    
    return autocorrelation_spins
