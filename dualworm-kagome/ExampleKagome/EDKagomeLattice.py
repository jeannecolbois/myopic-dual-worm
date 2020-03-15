
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import KagomeDrawing as kdraw
import GraphDrawing as gdw
import KagomeFunctions as kf
import DualwormFunctions as dw
import argparse


# In[ ]:

def getPositions(s_ijl,a = 2):
    pos = {} #empty dictionary
    for s, (i,j,l) in enumerate(s_ijl):
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
        pos[s] = (x,y)
    return pos


# In[ ]:

def spinHamiltonian():
    return 0


# In[ ]:

def unpackbits(x, num_bits):
    '''
    Thanks stackoverflow
    ...
    
    Unpacking the bits of an int with given number of bits
    '''

    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2**np.arange(num_bits).reshape([1, num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


# In[ ]:

def main(args):
    ## Initialisation
    L = args.L
    print('Lattice side size: ', L)

    (s_ijl, ijl_s) = kdraw.createspinsitetable(L)
    pos = getPositions(s_ijl, a = 2)

    plt.figure()
    plt.axis('equal')
    gdw.draw_nodes(pos, list(pos.keys()), c = "blue", s = 20)
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = False,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = False,
        left = False,
        right = False,
        labelleft = False)

    plt.figure()
    plt.axis('equal')
    kdraw.plot_kag_nodes(L,2)
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = False,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = False,
        left = False,
        right = False,
        labelleft = False)
    plt.show()

    [d_ijl, ijl_d, s_ijl, ijl_s, d_2s, s2_d, 
     d_nd, d_vd, d_wn, sidlist, didlist, c_ijl, ijl_c, c2s, csign] = dw.latticeinit(L)
    
    ## Hamiltonian
    J1 = args.J1
    J2 = args.J2
    J3 = args.J3
    J3st = J3
    J4 = args.J4
    h = args.h
    couplings = {'J1': J1, 'J2':J2, 'J3':J3, 'J3st':J3st, 'J4':J4}
    print("Couplings extracted: ", couplings)
    
    hamiltonian = dw.Hamiltonian(couplings,d_ijl, ijl_d, L)
    print("hamiltonian computed")
    
    minstate = np.zeros(9*L**2, dtype = "int8")
    
    energies = np.zeros(2**(9*L**2))
    minenergy = 0;
    for spinstateval in range(0,2**(9*L**2)):
        spinstate = unpackbits(np.array(spinstateval),9*L**2)*2 - 1
        state = np.zeros(len(d_ijl), dtype ='int8')
        for id_dim in range(len(d_ijl)):
            [id_s1, id_s2] = d_2s[id_dim]
            s1 = spinstate[id_s1]
            s2 = spinstate[id_s2]
            state[id_dim] = s1*s2
        energy = dw.compute_energy(hamiltonian, state)
        energies[spinstateval] = energy
        if energy < minenergy:
            minstate = spinstate
            minenergy = energy
    
    plt.figure()
    plt.axis('equal')
    kdraw.plot_kag_spinstate(minstate, ijl_s, L, 2, 'lightblue', 'blue', 'red', linewidth = 1)
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = False,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = False,
        left = False,
        right = False,
        labelleft = False)
    plt.show()
    return s_ijl, hamiltonian, minstate, minenergy, energies


# In[ ]:

if __name__ == "__main__":
    

    ### PARSING
    parser = argparse.ArgumentParser()

    parser.add_argument('--L', type = int, default = 2, help = 'Lattice side size')

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
    
    
    args = parser.parse_args()
    
    main(args)

