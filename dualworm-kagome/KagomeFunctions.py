
def compute_energy(hamiltonian, state, latsize):
    return dim.hamiltonian(hamiltonian, state)/latsize


# In[13]:


def conf_tester(state, latsize, L, d_ijl, ijl_d):
    #latsize = len(s_ijl)
    #Couplings
    J1 = 1
    J2 = 0.0
    J3 = 0.0
    J3st = J3
    J4 = 0.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    EJ1  = dim.hamiltonian(hamiltonian, state) /latsize


    #Couplings
    J1 = 0.0
    J2 = 1.0
    J3 = 0.0
    J3st = J3
    J4 = 0.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    EJ2  = dim.hamiltonian(hamiltonian, state) /latsize


    #Couplings
    J1 = 0.0
    J2 = 0.0
    J3 = 1.0
    J3st = J3
    J4 = 0.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    EJ3  = dim.hamiltonian(hamiltonian, state) /latsize

    #Couplings
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0
    J3st = J3
    J4 = 1.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    EJ4  = dim.hamiltonian(hamiltonian, state) /latsize

    print('J1 :', EJ1, '\nJ2 :', EJ2, '\nJ3 :', EJ3, '\nJ4 :', EJ4)

def corr_tester(state, latsize, L, d_ijl, ijl_d):
    #latsize = len(s_ijl)
    #Couplings
    J1 = 1
    J2 = 0.0
    J3 = 0.0
    J3st = J3
    J4 = 0.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    c1  = dim.hamiltonian(hamiltonian, state) /(2*latsize)


    #Couplings
    J1 = 0.0
    J2 = 1.0
    J3 = 0.0
    J3st = J3
    J4 = 0.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    c2 = dim.hamiltonian(hamiltonian, state) /(2*latsize)


    #Couplings
    J1 = 0.0
    J2 = 0.0
    J3 = 1.0
    J3st = J3
    J4 = 0.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    c3  = dim.hamiltonian(hamiltonian, state) /(2*latsize)

    #Couplings
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0
    J3st = J3
    J4 = 1.0

    #Hamiltonian
    hamiltonian = Hamiltonian(J1, J2, J3, J3st, J4, d_ijl, ijl_d, L)
    c4  = dim.hamiltonian(hamiltonian, state) /(2*latsize)

    return [c1, c2, c3, c4]

# In[14]:


def TruncatedNrj(state, n1, n2, Leff, power, s_pos, D, J1supp):
    '''
        Function computing the energy of the dipolar system by truncating in such a way that a spin doesn't interact with its images
    '''

    nrj = 0
    for s1 in range(len(s_pos)):
        for s2 in range(s1, len(s_pos), 1):
            (consider, dist) = pairseparation(s1, s2, s_pos, n1, n2, Leff,Leff)
            if consider and dist != 0:
                nrj += D/(dist**power)*state[s1]*state[s2]
            if np.abs(dist-1) < 1e-2:
                nrj += J1supp*D*state[s1]*state[s2]

    return nrj


# In[ ]:


def FiniteDistNrj(state, power, D, J1supp, neimax, distances, distances_spins):
    '''
        Function computing the energy of the dipolar system up to neimax neighbours (neimax included)
    '''

    nrj = NeiFromNeiToNrj(state, power, D, J1supp, 1, neimax, distances, distances_spins)

    return nrj


# In[ ]:


def NeiFromNeiToNrj(state, power, D, J1supp, neimin, neimax, distances, distances_spins):
    nrj = 0
    for n in range(neimin, neimax+1, 1):
         nrj += GivenNeiNrj(state, power, D, J1supp, n, distances, distances_spins)
    return nrj


# In[ ]:


def GivenNeiNrj(state, power, D, J1supp, nei, distances, distances_spins):
    nrj = 0
    dist = distances[nei]
    for (s1, s2) in distances_spins[dist]:
        nrj += D/(dist**power)*state[s1]*state[s2]
        if np.abs(dist-1) < 1e-2:
            nrj += J1supp*D*state[s1]*state[s2]

    return nrj


# In[15]:


from scipy.special import erfc
def EwaldSum(state, pairslist, s_pos, klat, D, alpha, S, J1supp):
    '''
        Function computing the energy of the dipolar system using the Ewald summation technique.
        /!\ alpha, klat and pairslist must have been chosen consistently or the result might be wrong
    '''
    constterm = - 2*D*alpha**3/(3*np.sqrt(pi))*len(state)

    print('constterm ok')
    realspaceterm = 0
    NNenergy = 0
    for (s1, s2), (r1, r2), dist in pairslist:
        if dist != 0:
            term1 = 2*alpha*np.exp(- alpha**2 * dist**2)/(np.sqrt(pi)*dist**2)
            term2 = erfc(alpha*dist)/(dist**3)
            realspaceterm += D*state[s1]*state[s2]*(term1 + term2)
        if np.abs(dist-1) < 1e-5:
            NNenergy += J1supp*D*state[s1]*state[s2]

    print('realspace ok')
    fourierspaceterm = 0
    for kvec in klat:
        k = np.linalg.norm(kvec)

        factor = 2*alpha/np.sqrt(pi) * np.exp(-k**2 / (4 * alpha**2)) - k*erfc(k/(2*alpha))
        ijabsum = 0
        for s1 in range(len(s_pos)):
            r1 = s_pos[s1]
            for s2 in range(len(s_pos)):
                r2 = s_pos[s2]
                ijabsum += state[s1] * state[s2] * np.exp( 1j * np.dot(kvec,(r2-r1)) )
        #for (s1, s2), (r1, r2), dist in pairslist:
        #    ijabsum += state[s1] * state[s2] * np.exp( 1j * np.dot(kvec,(r2-r1) ) )

        fourierspaceterm += factor*ijabsum

    print('fourier space ok')

    fourierspaceterm = (pi * D / S) * fourierspaceterm
    E = constterm + realspaceterm + fourierspaceterm + NNenergy
    issue = True
    if np.abs(np.imag(E)/np.real(E)) < 1e-16:
        issue = False

    return issue, np.real(E)


# In[16]:


#############################################################
                ## INITIALISING THE STATES ##
#############################################################


# In[17]:


def create_temperatures(nt_list, t_list):
    assert(len(t_list) == len(nt_list) + 1)
    nt = 0
    for nte in nt_list:
        nt += nte

    temp_states = np.zeros(nt)

    nt_start = 0
    for id_nt, nte in enumerate(nt_list):
        temp_states[nt_start: nte+nt_start] = np.linspace(t_list[id_nt], t_list[id_nt + 1], nte)
        nt_start += nte

    return np.unique(temp_states)


# In[ ]:


def create_log_temperatures(nt_list, t_list):
    assert(len(t_list) == len(nt_list) + 1)
    nt = 0
    for nte in nt_list:
        nt += nte

    temp_states = np.zeros(nt)

    ## Here, we have htat nt_list = [number between t0 and t1, number between t1 and t2, ...]

    nt_start = 0
    for id_nt, nte in enumerate(nt_list):
        temp_states[nt_start: nt_start + nte] = np.logspace(np.log10(t_list[id_nt]), np.log10(t_list[id_nt +1 ]), nte, endpoint=True)
        print(nt_start, nt_start+nte)
        nt_start +=nte
        
    return np.unique(temp_states)


# In[18]:


def candidate(s_ijl):
    spinstate = np.zeros(len(s_ijl))

    for s, (i, j, l) in enumerate(s_ijl):
        jp = i + 2*j
        ######### i = 0 ###############
        if i%4 == 0:
            if jp%12 == 0:
                if l == 0:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1

            if jp%12 == 2:
                if l == 1:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1
            if jp%12 == 4:
                if l == 1:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 6:
                if l == 2:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 8:
                if l == 1:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1
            if jp%12 == 10:
                if l == 1:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1

        ########## i = 1 ##############
        if i%4 == 1:
            if jp%12 == 1:
                if l == 1:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 3:
                if l == 2:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1
            if jp%12 == 5:
                if l == 1:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1
            if jp%12 == 7:
                if l == 2:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 9:
                if l == 1:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1
            if jp%12 == 11:
                if l == 2:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1


        ########## i = 2 ##############
        if i%4 == 2:
            if jp%12 == 0:
                spinstate[s] = -1
            if jp%12 == 2:
                if l == 0:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 4:
                if l == 0:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 6:
                spinstate[s] = -1
            if jp%12 == 8:
                if l == 2:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 10:
                if l == 2:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1


        ########## i = 3 ##############
        if i%4 == 3:
            if jp%12 == 1:
                if l == 0:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 3:
                if l == 1:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1
            if jp%12 == 5:
                if l == 0:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 7:
                if l == 1:
                    spinstate[s] = -1
                else:
                    spinstate[s] = 1
            if jp%12 == 9:
                if l == 0:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1
            if jp%12 == 11:
                if l == 1:
                    spinstate[s] = 1
                else:
                    spinstate[s] = -1

    return spinstate


# In[19]:


def state7shaped(s_ijl):
    spinstate7 = np.random.randint(0, 2, size=len(s_ijl))*2 - 1

    for s, (i, j, l) in enumerate(s_ijl):
        if (2*i + j)%4 == 0:
            if l == 0:
                spinstate7[s] = 1
            if l == 1:
                spinstate7[s] = -1
            if l == 2:
                spinstate7[s] = -1
        elif (2*i + j)%4 == 1:
            if l == 0:
                spinstate7[s] = 1
            if l == 1:
                spinstate7[s] = -1
            if l == 2:
                spinstate7[s] = 1
        elif (2*i + j)%4 == 2:
            if l == 0:
                spinstate7[s] = 1
            if l == 1:
                spinstate7[s] = 1
            if l == 2:
                spinstate7[s] = -1
        elif (2*i + j)%4 == 3:
            if l == 0:
                spinstate7[s] = 1
            if l == 1:
                spinstate7[s] = -1
            if l == 2:
                spinstate7[s] = -1

    return spinstate7


# In[20]:


def LargeJ2Init(spinstates, nt, s_ijl, same):
    version = np.random.randint(0,3)
    for t in range(nt):
        if not same:
            version = np.random.randint(0,3)
        sign = np.random.randint(0,2)*2-1
        if version == 0:
            ## Stripes of dimers
            for s, (i, j, l) in enumerate(s_ijl):
                if (2*i + j)%4 == 0:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = -sign
                elif (2*i + j)%4 == 1:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
                elif (2*i + j)%4 == 2:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                elif (2*i + j)%4 == 3:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = -sign
        if version == 1:
            # ferromagnetic spin bands with vertices of 4 spins
            for s, (i, j, l) in enumerate(s_ijl):
                if i%4 == 0:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
                if i%4 == 1:
                    if l == 0:
                        spinstates[t][s] = -sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 2:
                    if l == 0:
                        spinstates[t][s] = -sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 3:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
        if version == 2:
            #ferromagnetic spin bands with two flat pairs of two spins
            for s, (i, j, l) in enumerate(s_ijl):
                if i%4 == 0:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
                if i%4 == 1:
                    if l == 0:
                        spinstates[t][s] = -sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 2:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 3:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = -sign


# In[ ]:


def LargeJ2VersionInit(spinstates, nt, s_ijl, version):
    for t in range(nt):
        sign = np.random.randint(0,2)*2-1
        if version == 0:
            ## Stripes of dimers
            for s, (i, j, l) in enumerate(s_ijl):
                if (2*i + j)%4 == 0:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = -sign
                elif (2*i + j)%4 == 1:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
                elif (2*i + j)%4 == 2:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                elif (2*i + j)%4 == 3:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = -sign
        if version == 1:
            # ferromagnetic spin bands with vertices of 4 spins
            for s, (i, j, l) in enumerate(s_ijl):
                if i%4 == 0:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
                if i%4 == 1:
                    if l == 0:
                        spinstates[t][s] = -sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 2:
                    if l == 0:
                        spinstates[t][s] = -sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 3:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
        if version == 2:
            #ferromagnetic spin bands with two flat pairs of two spins
            for s, (i, j, l) in enumerate(s_ijl):
                if i%4 == 0:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = sign
                if i%4 == 1:
                    if l == 0:
                        spinstates[t][s] = -sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 2:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = sign
                    if l == 2:
                        spinstates[t][s] = -sign
                if i%4 == 3:
                    if l == 0:
                        spinstates[t][s] = sign
                    if l == 1:
                        spinstates[t][s] = -sign
                    if l == 2:
                        spinstates[t][s] = -sign


# In[21]:


def IntermediateInit(spinstates, nt, s_ijl):
    for t in range(nt):
        sign = np.random.randint(0,2)*2-1
        for s, (i, j, l) in enumerate(s_ijl):
            k = i + 2*j
            m = i - j
            if k%6 == 0:
                if m%6 == 0:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = - sign
                if m%6 == 3:
                    if l == 0:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = - sign

            if k%6 == 1:
                if m%6 == 1:
                    if l== 2:
                        spinstates[t][s] = - sign
                    else:
                        spinstates[t][s] = sign
                if m%6 == 4:
                    if l == 1:
                        spinstates[t][s] = - sign
                    else:
                        spinstates[t][s] = sign
            if k%6 == 2:
                if m%6 == 2:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = - sign
                if m%6 == 5:
                    if l == 0:
                        spinstates[t][s]= sign
                    else:
                        spinstates[t][s] = - sign
            if k%6 == 3:
                if m % 6 == 0:
                    if l == 2:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = - sign
                if m % 6 == 3:
                    spinstates[t][s] = sign

            if k%6 == 4:
                if m%6 == 1:
                    if l == 2:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = - sign
                if m%6 == 4:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = - sign

            if k%6 == 5:
                if m%6 == 2:
                    if l == 0:
                        spinstates[t][s] = - sign
                    else:
                        spinstates[t][s] = sign
                if m%6 == 5:
                    if l == 1:
                        spinstates[t][s] = - sign
                    else:
                        spinstates[t][s] = sign


# In[22]:


def LargeJ3Init(spinstates, nt, s_ijl):
    for t in range(nt):
        sign = np.random.randint(0,2)*2-1
        for s, (i, j, l) in enumerate(s_ijl):
            if (2*i+j)%6 == 0:
                if l == 2:
                    spinstates[t][s] = sign
                else:
                    spinstates[t][s] = - sign
            if (2*i+j)%6 == 1:
                if l == 2:
                    spinstates[t][s] = - sign
                else:
                    spinstates[t][s] = sign
            if (2*i+j)%6 == 2:
                if l == 2:
                    spinstates[t][s] = sign
                elif l == 1:
                    spinstates[t][s] = - sign
                elif l == 0:
                    spinstates[t][s] = np.random.randint(0, 2)*2 -1
            if (2*i+j)%6 == 3:
                if l == 1:
                    spinstates[t][s] = sign
                else:
                    spinstates[t][s] = - sign
            if (2*i+j)%6 == 4:
                if l == 1:
                    spinstates[t][s] = - sign
                else:
                    spinstates[t][s] = sign
            if (2*i+j)%6 == 5:
                if l == 1:
                    spinstates[t][s] = sign
                elif l == 2:
                    spinstates[t][s] = - sign
                elif l == 0:
                    spinstates[t][s] = np.random.randint(0, 2)*2 -1


# In[23]:


def DipolarToJ4Init(spinstates, nt, s_ijl):
    for t in range(nt):
        sign = np.random.randint(0,2)*2-1
        for s, (i, j, l) in enumerate(s_ijl):
            jp = i + 2*j
            ######### i = 0 ###############
            if i%4 == 0:
                if jp%12 == 0:
                    if l == 0:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign

                if jp%12 == 2:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign
                if jp%12 == 4:
                    if l == 1:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 6:
                    if l == 2:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 8:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign
                if jp%12 == 10:
                    if l == 1:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign

            ########## i = 1 ##############
            if i%4 == 1:
                if jp%12 == 1:
                    if l == 1:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 3:
                    if l == 2:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign
                if jp%12 == 5:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign
                if jp%12 == 7:
                    if l == 2:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 9:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign
                if jp%12 == 11:
                    if l == 2:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign


            ########## i = 2 ##############
            if i%4 == 2:
                if jp%12 == 0:
                    spinstates[t][s] = -sign
                if jp%12 == 2:
                    if l == 0:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 4:
                    if l == 0:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 6:
                    spinstates[t][s] = -sign
                if jp%12 == 8:
                    if l == 2:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 10:
                    if l == 2:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign


            ########## i = 3 ##############
            if i%4 == 3:
                if jp%12 == 1:
                    if l == 0:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 3:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign
                if jp%12 == 5:
                    if l == 0:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 7:
                    if l == 1:
                        spinstates[t][s] = -sign
                    else:
                        spinstates[t][s] = sign
                if jp%12 == 9:
                    if l == 0:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign
                if jp%12 == 11:
                    if l == 1:
                        spinstates[t][s] = sign
                    else:
                        spinstates[t][s] = -sign


# In[24]:


#nt = 50 # number of different temperatures = number of different states
def statesinit(nt, d_ijl, d_2s, s_ijl, hamiltonian, random = True, same = False):
    '''
       This function associates a dimer state to each dual bond. By default, this is done by first associating randomly a spin to
       each spin site and then map this onto a dimer state. If the initial state isn't random, it is what is believed to be the ground state
       of the problem.
       It returns the list of the states for each temperature and the energies.
    '''

    print('statesinit function called')
    #initialize the dimers
    states = [np.array([1 for i in range(len(d_ijl))], dtype='int32') for ignored in range(nt)]

    #initialize the spins randomly
    spinstates = [(np.random.randint(0, 2, size=len(s_ijl))*2 - 1) for i in range(nt)]

    if not random:
        print(' --> Non random initialisation')
        J1 = hamiltonian[0]
        if(len(hamiltonian) > 1):
            J2 = hamiltonian[1][0]
        else:
            J2 = 0
        if(len(hamiltonian )> 2):
            J3 = hamiltonian[2][0]
        else:
            J3 = 0
        if(len(hamiltonian )> 4):
            J4 = hamiltonian[4][0]
        else:
            J4 = 0

        # Initialise the states with the right family of ground states
        if J4 == 0:
            if J3 == 0:
                print('  >>> J1-J2 init')
                for s, (i, j, l) in enumerate(s_ijl):
                    #9 sublattices
                    subl = 3*((i + 2*j)%3) + l
                    for t in range(nt):
                        if subl in (0, 3, 6, 7):
                            spinstates[t][s] = 1
                        if subl in (2, 4, 5, 8):
                            spinstates[t][s] = -1
                        if subl == 1:
                            spinstates[t][s] = np.random.randint(0, 2) * 2 - 1
            if J2 != 0 and J3/J2 <= 0.5 and J3 > 0:
                print('  >>> LargeJ2 init')
                LargeJ2Init(spinstates, nt, s_ijl, same)
            if J2 != 0 and J3/J2 > 0.5 and J3/J2 <= 1:
                print('  >>> Intermediate init')
                IntermediateInit(spinstates, nt, s_ijl)
            if J2 != 0 and J3/J2 > 1:
                print('  >>> LargeJ3 init')
                LargeJ3Init(spinstates, nt, s_ijl)
            if J2 == 0:
                print('  >>> J1-J3 init')
                for s, (i, j, l) in enumerate(s_ijl):
                    #9 sublattices
                    subl = 3*((i + 2*j)%3) + l
                    for t in range(nt):
                        if subl in (0, 3, 5, 7):
                            spinstates[t][s] = -1
                        if subl in (1, 2, 6, 8):
                            spinstates[t][s] = 1
                        if subl == 4:
                            spinstates[t][s] = np.random.randint(0, 2) * 2 - 1
        else:
            print('  >>> D2J4 init')
            DipolarToJ4Init(spinstates, nt, s_ijl)

    #initialise the dimer state according to the spin state
    for t in range(nt):
        for id_dim in range(len(d_ijl)):
            [id_s1, id_s2] = d_2s[id_dim]
            s1 = spinstates[t][id_s1]
            s2 = spinstates[t][id_s2]
            if (s1 == s2):
                states[t][id_dim] = 1
            else:
                states[t][id_dim] = -1
    en_states = [dim.hamiltonian(hamiltonian, states[t]) for t in range(nt)] # energy computed via the function in c++

    return states, en_states


# In[25]:


def quickstatescheck(d_nd, d_vd, states):
    '''
        This function checks whether the dimer state makes sense
    '''
    ok = True
    for state in states:
        for d, nei in enumerate(d_nd):
            numd = (state[d] + 1) / 2.0
            for dim in nei:
                numd += (state[dim] + 1)/ 2.0
            if(numd%2 != 0): #entry site: 6-coordinated. Even number
                ok = False
        for d, nei in enumerate(d_vd):
            numd = (state[d] + 1) / 2.0
            for dim in nei:
                numd += (state[dim] + 1)/ 2.0
            if (numd%2 != 1): #vertex site: 3-coordinated, odd number
                ok = False
    return ok


# In[26]:


def statescheck(spinstates, states, d_2s):
    '''
        This function checks whether the dimer stateS and the spin stateS are compatible
    '''
    for spinstate, state in zip(spinstates, states):
        if len(onestatecheck(spinstate, state, d_2s)) != 0:
            return False
    return True


# In[27]:


def onestatecheck(spinstate, state, d_2s):
    '''
        This function checks whether the dimer state and the spin state are compatible
    '''
    mistakes = list()
    for id_d, d in enumerate(state):
        [id_s1, id_s2] = d_2s[id_d]
        s1 = spinstate[id_s1]
        s2 = spinstate[id_s2]
        if (s1 == s2 and d == -1):
            mistakes.append((id_d, id_s1, id_s2))
        if (s1 != s2 and d == 1):
            mistakes.append((id_d, id_s1, id_s2))
    return mistakes


# In[28]:


#############################################################
                ## STATES EVOLUTION ##
#############################################################


# In[29]:


### Define a function performing a series of MCS and then a series of swaps
# parallelism will be added later
from time import time
from threading import Thread
def mcs_swaps(nb, num_in_bin, iterworm, saveloops, temp_loops, temp_lenloops, check, statsfunctions, nt, stat_temps, hamiltonian, d_nd, d_vd, d_wn, d_2s, s2_d, sidlist, didlist, L, states, spinstates, en_states, beta_states, s_ijl, nitermax):
    '''
        This function creates the mcs class and then makes itermcs times the following:
            - ignored evolution for ignoredsteps (evolution during which the loops aren't registered)
            - considered evolution for consideredsteps (evolution during which the loops are registered)
            - swaps
    '''
    ############
    class mcs_withloops(Thread): #create a class inheritating from thread
        '''
            This class allows to make each thread evolve independently and save the loops
        '''
        def __init__(self, t, iterworm): #constructor
            Thread.__init__(self) # call to the thread constructor
            self.t = t # temperature index
            self.num = iterworm
            self.updates = list() # list of the updates which are created
            self.updatelengths = list() #list of length of updates

        def run(self):
            #with loops
            for _ in range(self.num):
                #update the state (notice that we work only with the DIMER state in the update code
                result = dim.manydualworms(hamiltonian, states[self.t], d_nd, d_vd, d_wn, beta_states[self.t], saveloops[self.t], nitermax) # the evolve function returns the difference in energy
                #update the energy
                en_states[self.t] += result[0]
                #statistics:
                #update loop
                self.updates.append(result[1])
                self.updatelengths.append(result[2])

    class mcs_withoutloops(Thread): #create a class inheritating from thread
        '''
            This class allows to make each thread evolve independently and not save the loops
        '''
        def __init__(self, t, iterworm): #constructor
            Thread.__init__(self) # call to the thread constructor
            self.t = t # temperature index
            self.num = iterworm
            self.updatelengths = list()

        def run(self):
            #without loops
            for _ in range(self.num):
                #update the state
                result = dim.manydualworms(hamiltonian, states[self.t], d_nd, d_vd, d_wn, beta_states[self.t], False, nitermax) # the evolve function returns the difference in energy
                #update the energy
                en_states[self.t] += result[0]

    class statistics(Thread):
        '''
            This class allows to update each spinstate independently and make the corresponding measurements
        '''
        def __init__(self, t, resid):
            Thread.__init__(self)
            self.t = t #temperature index
            self.resid = resid #index in the result list (eg if you only measure the smallest and highest temperature, resid = 1 for the highest temperature)

        def run(self):
            # Before doing the statistics, the spinstate must be updated
            spinstates[self.t] = onestate_dimers2spins(sidlist, didlist, L, states[self.t])
            # bin index = integer division of the iteration and the number of iterations in a bin
            b = it//num_in_bin
            for stat_id, stat_tuple in enumerate(statstables): #stat_id: index of the statistical function you're currently looking at
                func_per_site = statsfunctions[stat_id](states[self.t], en_states[self.t], spinstates[self.t], s_ijl) #evaluation depends on the temperature index
                stat_tuple[0][self.resid][b] += func_per_site / num_in_bin #storage depends on the result index
                stat_tuple[1][self.resid][b] += (func_per_site ** 2) / num_in_bin
    ##############

    # TABLES FOR STATISTICS
    statstables = [(np.zeros((len(stat_temps), nb)).tolist(), np.zeros((len(stat_temps), nb)).tolist()) for i in range(len(statsfunctions))] # stat_temps[i] -> to be able to have different lists of temperatures for different functions

    # ITERATE
    itermcs = nb * num_in_bin # number of mcs calls + swaps = number of bins * number of values taken into account in a bin
    t_join = 0
    t_spins = 0
    t_tempering = 0
    t_stat = 0
    swaps = [0 for t in range(nt)]
    for it in range(itermcs):
        t1 = time()
        ### MAKE ignoredsteps + consideredsteps
        thread_list = list()
        #create the threads
         # we make EVERYTHING evolve, including threads that we don't measure --> range(nt)
        for t in range(nt):
            if len(saveloops) == nt and saveloops[t] == 1:
                thread_list.append(mcs_withloops(t, iterworm))
            else:
                thread_list.append(mcs_withoutloops(t, iterworm))
        #start the threads
        for t in range(nt):
            thread_list[t].start()
        #wait for all threads to finish
        for t in range(nt):
            thread_list[t].join()
        #return the list of loops
        for t in range(nt):
            if len(saveloops) == nt and saveloops[t] == 1:
                temp_lenloops[t].append(thread_list[t].updatelengths)
                temp_loops[t].append(thread_list[t].updates)
        t2 = time()
        t_join += (t2-t1)/itermcs

        ##parallel tempering
        #take one pair after the other
        for t in range(nt-1, 0, -1):
            #throw a dice
            if (en_states[t] - en_states[t-1]) * (beta_states[t] - beta_states[t-1]) > 0: # if bigger than 0 flip for sure
                #flip accordingly
                #states:
                states[t - 1], states[t] = states[t], states[t - 1]
                #energy:
                en_states[t - 1], en_states[t] = en_states[t], en_states[t - 1]
                swaps[t] += 1
                swaps[t-1] += 1
            elif np.random.uniform() < np.exp((en_states[t] - en_states[t-1]) * (beta_states[t] - beta_states[t-1])): #else flip with a certain prob
                #flip accordingly
                #states:
                states[t - 1], states[t] = states[t], states[t - 1]
                #energy
                en_states[t - 1], en_states[t] = en_states[t], en_states[t - 1]
                swaps[t] += 1
                swaps[t-1] +=1
        t3 = time()
        t_tempering += (t3-t2)/itermcs

        ### STATISTICS UPDATE
        # bin index = integer division of the iteration and the number of iterations in a bin
        #   Before doing any measurement, the spinstate must be updated. But it is not necessary to update the spinstate
        #   if no measurement is performed, since the code runs on dimer configurations.
        #   Hence, we can feed the statistics threads with only the temperatures indices for which we are interested in
        #   the statistics.

        if len(statsfunctions) != 0 or check:
            thread_update_list = list()
            #create threads (NOT range(nt) but stat_temps because we only want to measure SOME states)
            for resid, t in enumerate(stat_temps):
                thread_update_list.append(statistics(t, resid))
            #start updating
            for resid, t in enumerate(stat_temps):
                thread_update_list[resid].start()
            #finish updating
            for resid, t in enumerate(stat_temps):
                thread_update_list[resid].join()
        t4 = time()
        t_spins += (t4-t3)/itermcs
    ### end For


    #verification:
    if len(statsfunctions) != 0 or check:
        for t in stat_temps:
            assert len(onestatecheck(spinstates[t], states[t], d_2s)) == 0, 'Loss of consistency at temperature index {0}'.format(t)
    #optimization info
    print('Time for building loops and saving them = {0}'.format(t_join))
    print('Time for tempering = {0}'.format(t_tempering))
    print('Time for mapping to spins + computing statistics= {0}'.format(t_spins))
    return statstables, swaps



# In[30]:


def onestateevolution(hamiltonian, state, d_nd, d_vd, d_wn, beta,  nitermax, iterworm):

    #true copy
    evolvingstate = np.copy(state)
    energy_differences = []
    updates = []
    updatelengths = []
    savedstates = [state]
    saveloops = True

    for _ in range(iterworm):
        result = dim.manydualworms(hamiltonian, evolvingstate, d_nd, d_vd, d_wn, beta, saveloops, nitermax)
        diff = result[0]
        if diff > 0:
            evolvingstate = np.copy(state)
        else:
            if len(result[1]) != 0:
                energy_differences.append(diff)
                updates.append(np.copy(result[1]))
                updatelengths.append(np.copy(result[2]))
                savedstates.append(np.copy(evolvingstate))

    return energy_differences, updates, updatelengths, savedstates


# In[31]:


def onestate_dimers2spins(sidlist, didlist, L, state):
    '''
        For a known state of the dual lattice (i.e. for each bond, is there or not a dimer), returns the corresponding
        spin state.
    '''
    #initialise the spins in a meaningless way
    spinstate = [0 for _ in range(len(sidlist))]
    #initialise the first spin randomly
    s_val =  np.random.randint(0, 2)*2-1
    s = spinstate[sidlist[0]] = s_val
    for it in range(0, len(sidlist)-1):
        db_id = didlist[it]
        spin_id = sidlist[it+1]
        s = spinstate[spin_id] = s * state[db_id]
    return spinstate


# In[32]:


def states_dimers2spins(sidlist, didlist, L, states):
    spinstates = []
    for state in states:
        spinstates.append(onestate_dimers2spins(sidlist, didlist, L, state))
    return spinstates


# In[33]:


#############################################################
                ## BASES FOR DRAWING ##
#############################################################


# In[34]:


from functools import lru_cache

@lru_cache(maxsize = None)
def graphdice(L, a):
    '''
        For the dice lattice:
        Returns two vertex <-> (i, j, l) tables, a table linking edge to the two corresponding vertices, as well as a dictionary giving the position (x,y) of each vertex
    '''
    #vertices table
    v_ijl = [(i, j, l) for i in range(-1, 2*L+1) for j in range(-1, 2*L+1) for l in range(3) if (i+j >= L-2) and (i+j <= 3*L - 1)]

    #vertices dictionary:
    ijl_v = {}
    for v, triplet in enumerate(v_ijl):
        ijl_v[triplet] = v

    #create the edges (v1, v2)
    #table of where to look at
    nv = [(0, -1, 2), (0, 0, 1), (0, 0, 2), (-1, 1, 1), (-1, 0, 2), (-1, 0, 1)]
    #edge -> vertex: l from 0 to 5 indicates the edge
    e_2v = [((i, j, 0),(i + nv[l][0], j + nv[l][1], nv[l][2])) for i in range(-1, 2*L+1) for j in range(-1, 2*L+1) for l in range(6) if (i+j >= L-2) and (i+j <= 3*L - 1)]
    e_2v = [(ijl_v[i, j, l], ijl_v[ni, nj, nl]) for ((i, j, l), (ni, nj, nl)) in e_2v if (ni, nj, nl) in v_ijl]
    #position
    pos = {} #empty dictionary
    for v, (i,j,l) in enumerate(v_ijl):
        if l == 0:
            x = a * (i + j / 2.0)
            y = a * j * np.sqrt(3) / 2.0
        if l == 1:
            x = a * (i + j / 2.0 + 1.0 / 2.0)
            y = a * (j * np.sqrt(3) / 2.0 - 1.0 / (2.0 * np.sqrt(3.0)))
        if l == 2:
            x = a * (i + j / 2.0 + 1.0 / 2.0)
            y = a * (j * np.sqrt(3) / 2.0 + 1.0/ (2.0 * np.sqrt(3.0)))
        pos[v] = (x,y)
    return v_ijl, ijl_v, e_2v, pos


# In[35]:


def reducedgraphdice(L, a, d_ijl, v_ijl, ijl_v):
    #table of where to look at
    nv = [(0, -1, 2), (0, 0, 1), (0, 0, 2), (-1, 1, 1), (-1, 0, 2), (-1, 0, 1)]
    #dual bond -> vertex: l from 0 to 5 indicates the edge
    d_2v = [((i, j, 0),(i + nv[l][0], j + nv[l][1], nv[l][2])) for (i, j, l) in d_ijl]
    d_2v = [(ijl_v[i, j, l], ijl_v[fixbc(ni, nj, nl, L)]) for ((i, j, l), (ni, nj, nl)) in d_2v]

    v2_d = {}
    for d, (v1,v2) in enumerate(d_2v):
        v2_d[v1,v2] = d
        v2_d[v2,v1] = d

    return d_2v, v2_d


# In[36]:


#Function creating a list of vertices corresponding to the kagome lattice
from functools import lru_cache

@lru_cache(maxsize = None)
def graphkag(L, a):
    '''
        For the kagomé lattice:
        Returns two vertex <-> (i, j, l) tables, a table linking edge to the two corresponding vertices, as well as a dictionary giving the position (x,y) of each vertex
    '''
    #vertices table
    sv_ijl = [(i, j, l) for i in range(-1, 2*L+1) for j in range(-1, 2*L+1) for l in range(3) if (i+j >= L-2) and (i+j <= 3*L - 1)]

    #vertices dictionary:
    ijl_sv = {}
    for sv, triplet in enumerate(sv_ijl):
        ijl_sv[triplet] = sv

    #create the edges (sv1, sv2)
    #table of where to look at
    nv = [[(0, 0, 1), (1, 0, 2)],[(-1, 1, 0), (1, 0, 2)], [(0, 0, 1), (-1, 1, 0)]]
    #edge -> vertex: l from 0 to 5 indicates the edge
    e_2sv = [((i, j, l),(i + nv[l][u][0], j + nv[l][u][1], nv[l][u][2])) for i in range(-1, 2*L+1) for j in range(-1, 2*L+1) for l in range(3) for u in range(2) if (i+j >= L-2) and (i+j <= 3*L - 1)]
    e_2sv = [(ijl_sv[i, j, l], ijl_sv[ni, nj, nl]) for ((i, j, l), (ni, nj, nl)) in e_2sv if (ni, nj, nl) in sv_ijl]
    #position
    pos = {} #empty dictionary
    for sv, (i,j,l) in enumerate(sv_ijl):
        x = a * (i + j / 2.0)
        y = a * j * np.sqrt(3) / 2.0
        if l == 0:
            x += a / 2.0
        if l == 1:
            x += a / 4.0
            y += a * np.sqrt(3) / 4.0
        if l == 2:
            x -= a / 4.0
            y += a * np.sqrt(3) / 4.0
        pos[sv] = (x,y)
    return sv_ijl, ijl_sv, e_2sv, pos


# In[37]:


def reducedgraphkag(L, s_ijl, ijl_s):
    '''
        For the kagome lattice:
        returns only one position for each spin (i,j,l) location
    '''
    #position
    s_pos = {} #empty dictionary
    ijl_pos = {}
    for s, (i,j,l) in enumerate(s_ijl):
        x = (2*i + j)
        y = j * np.sqrt(3)
        if l == 0:
            x += 1
        if l == 1:
            x += 0.5
            y +=np.sqrt(3) / 2.0
        if l == 2:
            x -= 0.5
            y += np.sqrt(3) / 2.0
        s_pos[s] = np.array((x,y))
        ijl_pos[s_ijl[s]] = np.array((x,y))
    return s_pos, ijl_pos


# In[38]:


def superlattice(L):
    n1 = np.array([np.sqrt(3)/2, -1/2])
    n2 = np.array([np.sqrt(3)/2, 1/2])
    Leff = 2*np.sqrt(3)*L
    S = np.sqrt(3)/2 * Leff**2

    return n1, n2, Leff, S


# In[39]:


def reciprocallattice():
    #Defining the reciprocal lattice of the superlattice
    b1 = 2*pi/(3*L)*np.array([1/2, -np.sqrt(3)/2])
    b2 = 2*pi/(3*L)*np.array([1/2, np.sqrt(3)/2])

    return b1, b2


# In[40]:


def kvectorslist(b1, b2, kmax, nok0):
    '''
        Returns a list of k vectors given the reciprocal basis (b1, b2) and a vector kmax.
        kmax is chosen in such a way that the edge don't contribute, so it doesn't really matter what shape we
        give the cell
    '''
    klat = []
    kidmax = int(kmax/(np.sqrt(2)*np.linalg.norm(b1)))
    for kx in range(-kidmax, kidmax+1, 1):
        for ky in range(-kidmax, kidmax+1, 1):
            if nok0:
                if not(kx == 0 and ky == 0):
                    kvec = kx*b1 + ky*b2
                    klat.append(kvec)
            else:
                kvec = kx*b1 + ky*b2
                klat.append(kvec)
    return klat


# In[41]:


def pairseparation(s1, s2, s_pos, n1, n2, Leff, distmax):
    '''
        Given two spin sites s1 and s2, this function returns the *minimal distance* between the two sites
        (considering pbc) and tells if it is less than Leff/2
    '''
    #What we will do is just list all the six possible positions for the spin s1 and take the minimal distance
    pos1 = s_pos[s1]
    pos2 = s_pos[s2]
    pos2list = []
    listnei = [(0, 0), (0, 1), (1, 0), (-1, 1), (-1, 0), (0, -1),(1, -1)]

    for nei in listnei:
        pos2list.append(pos2 + nei[0]*Leff*n1 + nei[1]*Leff*n2)

    distmin = 10*Leff
    lessthanmax = False
    for pos in pos2list:
        dist = np.linalg.norm(pos1 - pos)
        if dist < distmin:
            distmin = dist

    if distmin < min(Leff/2, distmax):
        lessthanmax = True

    return lessthanmax, distmin


# In[42]:


def sitepairslist(srefs, s_pos, n1, n2, Leff, distmax):
    '''
        For a given structure, this function returns a table containing, for each pair ({i1, j1, l1}, {i2, j2, l2}) at distance less than Leff/2, the corresponding distance R and the *indices* s1 and s2 of the spins in positions these positions.
        
        We only consider couples containing spins in srefs.
        
        
        It returns as well an ordered list of the distances and a dictionary associating each distance to a set of spins.
    '''
    
     # for each distance, we get the various spins that are at this distance from a given spin index

    pairs = []
    distmin = Leff
    
   
    for s1 in srefs:
        for s2 in range(len(s_pos)):
            (consider, dist) = pairseparation(s1, s2, s_pos, n1, n2, Leff, distmax)
            if consider:
                if dist < distmin:
                    distmin = dist
                
                pairs.append(((s1, s2), dist))
                
    distances = []
    distances_spins = {}
    for (spair, dist) in pairs:
        dist = np.round(dist, 4)
        if dist != 0:
            if dist in distances:
                distances_spins[dist].append(spair)
            else:
                distances.append(dist)
                distances_spins[dist] = [spair]

    return pairs, sorted(distances), distances_spins


# In[ ]:


def dist_sitepairs(s_pos,  n1, n2, Leff):
    pairs = sitepairslist(s_pos, n1, n2, Leff)
    distances = []
    distances_spins = {}
    for (spair, spospair, dist) in pairs:
        dist = np.round(dist, 4)
        if dist in distances:
            distances_spins[dist].append(spair)
        else:
            distances.append(dist)
            distances_spins[dist] = [spair]

    return sorted(distances), distances_spins


# In[43]:

def KagomeNearestNeighboursLists(L, distmax):
    '''
        Returns a list of distances between sites (smaller than distmax) with respect to the 3 reference sites, a dictionary of pairs of sites at a given distance and a list of the nearest neighbour pairs associated with a given site and distance.
    '''
    #dimer table and dictionary:
    (d_ijl, ijl_d) = createdualtable(L)
    #spin table and dictionary
    (s_ijl, ijl_s) = createspinsitetable(L)
    #two spins surrounding each dimer
    (d_2s, s2_d) = dualbondspinsitelinks(d_ijl, ijl_s, L)
    #dimer-dimer connection through entry sites
    d_nd = nsitesconnections(d_ijl, ijl_d)
    #dimer-dimer connection through vertex sites
    d_vd = vsitesconnections(d_ijl, ijl_d, L)
    #for each dimer, is it takeing into account in winding number 1 or 2?
    d_wn = windingtable(d_ijl, L)
    #list of spin indices and dimer indices for the loop allowing to update the spin state
    (sidlist, didlist) = spins_dimers_for_update(s_ijl, ijl_s, s2_d, L)
    
    
    #graph
    (s_pos, ijl_pos) = reducedgraphkag(L, s_ijl, ijl_s)
    pos = list(s_pos.values())
    pos = [list(np.round(posval, 4)) for posval in pos]
    #initialise the superlattice
    (n1, n2, Leff, S) = superlattice(L)
    
    # getting the list of pairs that we're interested in, 
    srefs = [ijl_s[(L,L,0)], ijl_s[(L,L,1)], ijl_s[(L,L,2)]]
    pairs, distances, distances_spins = sitepairslist(srefs, s_pos, n1,n2,Leff,distmax)
    
    NNList = [[[] for i in range(len(distances))] for j in range(len(srefs))]
    
    for i in range(len(distances)):
        for pair in distances_spins[distances[i]]:
            for j in range(len(srefs)):
                if srefs[j] in pair:
                    NNList[j][i].append(pair)

    # correct the neighbour lists elements that can cause trouble
    
    distances.insert(3, distances[2])
    distances.insert(7, distances[6])
    for j in range(len(srefs)):
        NNList3_0 = []
        NNList3_1 = []
        for (s1,s2) in NNList[j][2]:
            halfway = np.round((s_pos[s1] + s_pos[s2])/2, 4)
            if list(halfway) in pos:
                NNList3_0.append((s1,s2))
            else:
                NNList3_1.append((s1,s2))
        
        NNList6_0 = []
        NNList6_1 = []
        for (s1,s2) in NNList[j][5]:
            halfway = np.round((s_pos[s1] + s_pos[s2])/2, 4)
            if list(halfway) in pos:
                NNList6_0.append((s1,s2))
            else:
                NNList6_1.append((s1,s2))
        
            # replacing in NNList
        NNList[j][2] =  NNList3_0
        NNList[j].insert(3, NNList3_1)
        NNList[j][6] = NNList6_0
        NNList[j].insert(7, NNList6_1)
        
    return distances, distances_spins, NNList, s_pos, srefs


def neimax_distmax(s_pos, neimax):
    '''
        Function returning the distance corresponding to the neimaxth neighbour
    '''
    dist = 0

    #create the list of distances
    distlist = []
    for s1 in range(len(s_pos)):
        r1 = s_pos[s1]
        for s2 in range(len(s_pos)):
            r2 = s_pos[s2]
            dist = np.round(np.linalg.norm(r1-r2),3)
            if dist not in distlist:
                distlist.append(dist)

    #sort the list of distances
    distlist.sort()

    if len(distlist) <= neimax :
        dist = -1
    else:
        dist = distlist[neimax]

    return dist


# In[44]:


#############################################################
                ## DICE LATTICE STATE ##
#############################################################


# In[45]:


# FUCTION LINKING THE EDGE OF THE GRAPH TO THE CORRESPONDING DIMER STATE
def edge2dimer(L, a, d_ijl, v_ijl, ijl_v, e_2v):
    (d_2v, v2_d) = reducedgraphdice(L,a, d_ijl, v_ijl, ijl_v) #for the two vertices in the reduced bc
    e_d = list()
    for e, (v1, v2) in enumerate(e_2v):
        (i1, j1, l1) = v_ijl[v1]
        (i2, j2, l2) = v_ijl[v2]
        v1 = ijl_v[fixbc(i1, j1, l1, L)]
        v2 = ijl_v[fixbc(i2, j2, l2, L)]
        d = v2_d[v1, v2]
        e_d.append(d)
    return e_d


# In[46]:


#############################################################
                ## KAGOME LATTICE STATE ##
#############################################################


# In[47]:


def spinvertex2spin(L,a, ijl_s, sv_ijl):
    '''
        Given a list of spin vertices, associates to each one the corresponding spin
    '''
    sv_s = list()
    for sv, (i, j, l) in enumerate(sv_ijl):
        (ni, nj, nl) = fixbc(i, j, l, L)
        s = ijl_s[ni, nj, nl]
        sv_s.append(s)

    return sv_s
