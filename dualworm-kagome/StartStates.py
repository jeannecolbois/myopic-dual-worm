
# coding: utf-8

# In[ ]:

from DualwormFunctions import compute_energy
import numpy as np


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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
    en_states = [compute_energy(hamiltonian, states[t]) for t in range(nt)] # energy computed via the function in c++

    return states, en_states

