
# coding: utf-8

# This is a few functions which will help with sorting state for the MFU

# In[ ]:

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


# In[ ]:

import numpy as np
import DualwormFunctions as dw


# In[ ]:

def StatesCompare(state1, state2, ratio, d_w1, d_w2, d_ijl, **kwargs):
    '''
        This function compares two states and decides whether they belong to the same family
    '''
    print("Comparing...")
    # 1 - Make the difference between the two dimer states
    diffstate = state1-state2
    
    # 2 - Check if the two states are the same
    same = False
    if np.all(diffstate==0):
        same = True
        
    if not same:
        # 3 - Compute the winding number
        # 3.1 - Find all the dual bonds where the state has flipped
        df = np.nonzero(diffstate)
        df = df[0]
        
        print("length of df = ", df.size)
        # 3.2 - If the number of flipped dimers is high as compared to d_ijl,
        # then we consider the move to be non-local (to avoid spending too much time
        # finding out the winding number)
        if df.size/len(d_ijl) > ratio:
            samefamily = False
        else:
            print("Need detailed comparison")
            # 3.3 find the dimers contributing to the winding numbers: 
            w1array = np.intersect1d(df, d_w1)
            w2array = np.intersect1d(df, d_w2)
            
            print("w1 array:", w1array)
            print("w2 array:", w2array)
            
            if w1array.size == 0 and w2array.size == 0:
                samefamily = True
            else: # if some of it is non-zero, then we do need to check.
                samefamily = False
            # FOR NOW WE DO IT SUPER SIMPLE; PROPER WAY WOULD BE:
            
            #if len(w1array) % 2 == 1 or len(w2array) % 2 == 1:
            #    samefamily = False
            #else:
            #    # 3.4 - Actually go through the loopS to determine
            #    # whether the move is local or not
            #    df2empty = np.copy(df) # true copy
            #    notempty = True
            #    w1 = 0
            #    w2 = 0
            #    while notempty:
            #         # Go through the loop and check the direction of the crossing each time.        
                    
    else:
        samefamily = True
    
    return same, samefamily


# In[ ]:

def FamiliesFromStates(hamiltonian,liststates,
                       gsenergy, listspinstates,
                       d_wn, latsize, ratio, d_ijl,
                       **kwargs):
    '''
        From a list of states, this function creates families of similar state
        Returns:
        - families : list of families, which are list of states indices
        - spinsfamilies: list of spin families, which are list of spinstates
        (corresponding to the state indices)
    '''
    
    d_w1, d_w2 = dw.winding1and2(d_wn)
    families = []
    for stateid in range(liststates.shape[0]):
        print("stateid = ", stateid)
        state = liststates[stateid]
        # 1 - check that the state is in the ground state
        check = dw.check_energy(hamiltonian, state, gsenergy, latsize = latsize)
        
        # 2 - If it is, compare it against all the families
        if check:
            if len(families) != 0:
                # compare to the pre-existing families
                notyet = True
                index = 0
                
                # first round: check if it's the same state as one in an existing family
                print("First round")
                while notyet and index < len(families):    
                    diffstate = state-liststates[families[index][0]]
                    if np.all(diffstate==0):
                        same = True
                        notyet = False
                    index += 1
                        
                
                # second round: if not, check more carefully what's going on
                index = 0
                if notyet:
                    print("Second round")
                while notyet:
                    print("Index: ", index)
                    if index < len(families):
                        [same, samefamily] =                        StatesCompare(state, liststates[families[index][0]], ratio,
                                      d_w1, d_w2, d_ijl, **kwargs)
                        if same:
                            print("Somehow missed that it's the same the first time...")
                        elif (not same) and samefamily:
                            families[index].append(stateid)
                            notyet = False
                        else:
                            index += 1
                    else:
                        #the state does not belong to a pre-existing family
                        families.append([stateid])
                        notyet = False
            else: # if families is empty
                families.append([stateid])
        else:
            print("Not in the gs")
    # 3 - Return the families
    spinfamilies = []
    for family in families:
        spinfamily = []
        for stateid in family:
            spinfamily.append(listspinstates[stateid])
        spinfamilies.append(spinfamily)
        
    print("Done!")
    return families, spinfamilies
        

