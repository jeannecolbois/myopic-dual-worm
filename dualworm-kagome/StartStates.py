
# coding: utf-8

# In[ ]:

from DualwormFunctions import compute_energy
import numpy as np


# In[ ]:

def fixbc(i, j, l, L):
    '''
        For a lattice side size L, this function handles the periodic boundary conditions by returning the corresponding
        value of i, j, l if they match a point which is just outside the borders of the considered cell.
    '''
    if i == 2*L : # bottom right mapped to top left
        i = 0
        j += L
    if j == 2*L: # top mapped to bottom
        i += L
        j = 0
    if i+j == L-2: # bottom left mapped to top right
        i += L
        j += L
    if i+j == 3*L-1: # top right mapped to bottom left
        i -= L
        j -= L
    if j == -1: # bottom mapped to top
        i -= L
        j = 2*L-1
    if i == -1: # top left mapped to bottom right
        i = 2*L-1
        j -= L
    return (i, j, l)


# In[ ]:

def magnetisedInit(spinstates, nt, nh, s_ijl, hfields, same):
   
    if same:
        
        version = np.random.randint(0,2)
        
        if version:
            magnetisedInitStripes(spinstates, nt, nh, s_ijl, hfields, same)
        else:
            magnetisedInitMaxFlip(spinstates, nt, nh, s_ijl, hfields, same)
            
    else:
        print("Magnetised init")
        versions = [np.random.randint(0,2) for w in range(nt*nh)]
        
        for t in range(nt*nh):
            for h in range(nh):
                w = t*nh+h
                newspinstates = [(np.random.randint(0, 2, size=len(s_ijl), dtype = 'int8')*2 - 1)]
                if versions[w]:
                    magnetisedInitStripes(newspinstates, 1, 1, s_ijl, [hfields[h]], True)
                else:
                    magnetisedInitMaxFlip(newspinstates, 1, 1, s_ijl, [hfields[h]], True)
                
                spinstates[w] = newspinstates[0]


# In[ ]:

def magnetisedInitStripes(spinstates, nt, nh,
                          s_ijl, hfields, same):
   
    if same:
        
        if hfields[0] == 0:
            sign = np.random.randint(0,2, dtype = 'int8')*2-1
        else:
            sign = np.sign(hfields[0], dtype = 'int8')
        
        version = np.random.randint(0,3, dtype = 'int8')
        
        spinstate = sign*np.ones(len(s_ijl))
        
        for s, (i,j,l) in enumerate(s_ijl):
            if l == version:
                spinstate[s] = -sign
                
        for w in range(nt*nh):
            spinstates[w] = np.copy(spinstate.astype('int8')) # to make sure that it is DIFFERENT states
        
    else:
        print("Magnetised init - stripes")
        versions = [np.random.randint(0,3) for w in range(nt*nh)]
        
        signs = [np.sign(h) for t in range(nt) for h in hfields]
        
        for w in range(nt*nh):
            if signs[w] == 0:
                signs[w] = np.random.randint(0,2, dtype = 'int8')*2-1
            for s, (i,j,l) in enumerate(s_ijl):
                if l == versions[w]:
                    spinstates[w][s] = -signs[w]
                else:
                    spinstates[w][s] = signs[w]


# In[ ]:

def magnetisedInitMaxFlip(spinstates, nt, nh,
                          s_ijl, hfields, same):
    if same:
        if hfields[0] == 0:
            sign = np.random.randint(0,2)*2-1
        else:
            sign = np.sign(hfields[0])
            
        version = np.random.randint(0,3)
        
        spinstate = sign*np.ones(len(s_ijl))
        
        for s, (i,j,l) in enumerate(s_ijl):
            loc = (i - j + version)%3
            if loc == 0:
                if l == 1:
                    spinstate[s] = -sign
                else:
                    spinstate[s] = sign
            elif loc == 1:
                spinstate[s] = sign
            elif loc == 2:
                if l == 1:
                    spinstate[s] = sign
                else:
                    spinstate[s] = - sign
        for w in range(nt*nh):
            spinstates[w] = np.copy(spinstate)
            
    else:
        print("Magnetised init - max flip")
        versions = [np.random.randint(0,3) for w in range(nt*nh)]
        
        signs = [np.sign(h,dtype = 'int8') for t in range(nt) for h in hfields]
        
        for w in range(nt*nh):
            if signs[w] == 0:
                signs[w] = np.random.randint(0,2,dtype = 'int8')*2-1
            for s, (i,j,l) in enumerate(s_ijl):
                loc = (i - j + versions[w])%3
                if loc == 0:
                    if l == 1:
                        spinstates[w][s] = -signs[w]
                    else:
                        spinstates[w][s] = signs[w]
                elif loc == 1:
                    spinstates[w][s] = signs[w]
                elif loc == 2:
                    if l == 1:
                        spinstates[w][s] = signs[w]
                    else:
                        spinstates[w][s] = - signs[w]
                    
                        


# In[ ]:

def magnetisedInit06(spinstates, nt, nh, s_ijl, hfields, same):
   
    if same:
        if hfields[0] == 0:
            sign = np.random.randint(0,2, dtype = 'int8')*2-1
        else:
            sign = np.sign(hfields[0], dtype = 'int8')
        
        version = np.random.randint(0,3)
        
        spinstate = sign*np.ones(len(s_ijl))
        
        for s, (i,j,l) in enumerate(s_ijl):
            loc = (i - j + version)%5
            if loc == 0 or loc == 2:
                if l == 1:
                    spinstate[s] = -sign
                else:
                    spinstate[s] = sign
            elif loc == 1 or loc == 3:
                spinstate[s] = sign
            elif loc == 4:
                if l == 0:
                    spinstate[s] = sign
                else:
                    spinstate[s] = -sign
                    
        for t in range(nt*nh):
            for h in range(nh):
                w = t*nh+h
                spinstates[w] = np.copy(spinstate) # to make sure that it is DIFFERENT states
        
    else:
        versions = [np.random.randint(0,3) for w in range(nt*nh)]
        signs = [np.sign(h,dtype = 'int8') for t in range(nt) for h in hfields]
        
        for t in range(nt*nh):
            for h in range(nh):
                w = t*nh+h
                for s, (i,j,l) in enumerate(s_ijl):
                    loc = (i - j + versions[w])%5

                    if loc == 0 or loc == 2:
                        if l == 1:
                            spinstates[w][s] = -signs[w]
                        else:
                            spinstates[w][s] = signs[w]
                    elif loc == 1 or loc == 3:
                        spinstates[w][s] = signs[w]
                    elif loc == 4:
                        if l == 0:
                            spinstates[w][s] = signs[w]
                        else:
                            spinstates[w][s] = -signs[w]


# In[ ]:

def J1J2Init(spinstates, nt, nh, s_ijl, same):
    if same:
        #print("In same (J1J2)")
        version = np.random.randint(0,3)
        sign = np.random.randint(0,2, dtype = 'int8')*2 -1
        
        spinstate = np.ones(len(s_ijl), dtype = 'int8')
        for s, (i,j,l) in enumerate(s_ijl):
            if l == version:
                spinstate[s] = sign
            if l == (version+1)%3:
                spinstate[s] = -sign
            if l == (version+2)%3:
                spinstate[s] = np.random.randint(0,2, dtype = 'int8')*2-1
        
        for t in range(nt*nh):
            for h in range(nh):
                w = t*nh+h
                spinstates[w] = np.copy(spinstate.astype('int8'))
    else:
        #print("In Differents (J1J2)")
        versions = [np.random.randint(0,3) for w in range(nt*nh)]
        signs = [np.random.randint(0,2, dtype = 'int8')*2 -1 for w in range(nt*nh)]
        #print("versions and signs generated (J1J2)")
        for t in range(nt*nh):
            for h in range(nh):
                w = t*nh+h
                for s, (i, j, l) in enumerate(s_ijl):
                    if l == versions[w]:
                        spinstates[w][s] = signs[w]
                    if l == (versions[w] + 1)%3:
                        spinstates[w][s] = -signs[w]
                    if l == (versions[w] + 2)%3:
                        spinstates[w][s] = np.random.randint(0,2, dtype = 'int8')*2 - 1


# In[ ]:

def J1J3InitOneState(spinstate, s_ijl, version, sign, shift, rot):
    if version == 0:
        if rot == 0:
            for s, (i,j,l) in enumerate(s_ijl):
                loc = (i+2*j+shift)%3
                if loc == 0:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if loc == 1:
                    if l == 1:
                        spinstate[s] = - sign
                    else:
                        spinstate[s] = np.random.randint(0,2, dtype = 'int8')*2 -1
                if loc == 2:
                    if l == 1:
                        spinstate[s] = np.random.randint(0,2, dtype = 'int8')*2 - 1
                    else:
                        spinstate[s] = sign
                        
    spinstate = spinstate.astype('int8')


# In[ ]:

def J1J3Init(spinstates, nt, nh, s_ijl, same):
    if same:
        sign = np.random.randint(0,2, dtype = 'int8')*2 -1
        shift = np.random.randint(0,3)
        spinstate = np.ones(len(s_ijl))
        version = 0
        rot = 0
        
        J1J3InitOneState(spinstate,s_ijl, version, sign, shift, rot)
        spinstates = [np.copy(spinstate.astype('int8')) for w in range(nt*nh)]
    else:
        versions = [0 for w in range(nt*nh)]
        signs = [np.random.randint(0,2, dtype = 'int8')*2-1 for w in range(nt*nh)]
        shifts = [np.random.randint(0,4) for w in range(nt*nh)]
        rots = [0 for w in range(nt*nh)]
        
        for t in range(nt*nh):
            J1J3InitOneState(spinstates[t], s_ijl, versions[t], signs[t], shifts[t], rots[t])


# In[ ]:

def LargeJ2InitOneState(spinstate, s_ijl, version, sign, shift, rot):
    if version == 0:
        ## Stripes of dimers
        if rot == 0:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (2*i + j + shift)%4 
                if  loc == 0:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = -sign
                elif loc == 1:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                elif loc == 2:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                elif loc == 3:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = -sign
        elif rot == 1:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i + shift - j)%4 
                if loc == 0:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                elif loc == 1:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = -sign
                elif loc == 2:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                elif loc == 3:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
        elif rot == 2:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i + shift +2*j)%4 
                if loc == 0:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                elif loc == 1:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                elif loc == 2:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                elif loc == 3:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = -sign
    elif version == 1:
            # ferromagnetic spin bands with vertices of 4 spins
        if rot == 0:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i+shift)%4
                if loc == 0:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc == 1:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc == 2:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc == 3:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
        elif rot == 1:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i+shift +j)%4
                if loc == 0:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc == 1:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc == 2:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc == 3:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
        elif rot == 2:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (j+shift)%4
                if loc == 0:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc == 1:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc == 2:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc == 3:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = sign
    elif version == 2:
        if rot == 0:
            #ferromagnetic spin bands with two flat pairs of two spins
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i+shift)%4
                if loc%4 == 0:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc%4 == 1:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc%4 == 2:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc%4 == 3:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = -sign
        elif rot == 1:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i+j+shift)%4
                if loc%4 == 0:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc%4 == 1:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc%4 == 2:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = sign
                if loc%4 == 3:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = -sign
        elif rot == 2:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (j+shift)%4
                if loc%4 == 0:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc%4 == 1:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = -sign
                if loc%4 == 2:
                    if l == 0:
                        spinstate[s] = sign
                    if l == 1:
                        spinstate[s] = -sign
                    if l == 2:
                        spinstate[s] = sign
                if loc%4 == 3:
                    if l == 0:
                        spinstate[s] = -sign
                    if l == 1:
                        spinstate[s] = sign
                    if l == 2:
                        spinstate[s] = sign
                        
    spinstate = spinstate.astype('int8')


# In[ ]:

def LargeJ2Init(spinstates, nt, nh, s_ijl, same):
    if same:
        version = np.random.randint(0,3)
        sign = np.random.randint(0,2, dtype = 'int8')*2-1
        shift = np.random.randint(0,4)
        rot = np.random.randint(0,3)
        
        spinstate = np.zeros(len(s_ijl))
        LargeJ2InitOneState(spinstate, s_ijl, version, sign, shift, rot)
        spinstates = [np.copy(spinstate) for w in range(nt*nh)]
    else:
        versions = [np.random.randint(0,3) for w in range(nt*nh)]
        signs = [np.random.randint(0,2, dtype = 'int8')*2-1 for w in range(nt*nh)]
        shifts = [np.random.randint(0,4) for w in range(nt*nh)]
        rots = [np.random.randint(0,3) for w in range(nt*nh)]
        
        for t in range(nt*nh):
            LargeJ2InitOneState(spinstates[t], s_ijl, versions[t], signs[t], shifts[t], rots[t])


# In[ ]:

def IntermediateInitOneState(spinstate, s_ijl,version, sign, shift, rot):
    # Prepare cells to flip
    L = int(np.sqrt(len(s_ijl)//9))
    listtoflip = []
    for flipid in range(len(s_ijl)//(9*3)):
        sid = np.random.randint(0,len(s_ijl))
        (i,j,l) = s_ijl[sid]
        (i,j,l) = (i,j,2);
        if not listtoflip: # only if no flip
            if version == 0:
                if rot == 0:
                    k = (i + 2*j +shift)
                    m = (i - j+shift)
                    if k%6 ==5 and m%6 == 5:
                        localflip = [fixbc(i,j,2,L), fixbc(i-2, j+1,0,L), fixbc(i-2, j, 1,L),
                                     fixbc(i-1, j-1,2,L), fixbc(i-1,j-1,0,L), fixbc(i,j-1, 1,L)]
                        listtoflip.append(localflip)
                        
    print("List to flip:", listtoflip)
    for s, (i, j, l) in enumerate(s_ijl):
        if not listtoflip:
            flipfactor = 1
        else:
            flipfactors = [-1 for localflip in listtoflip if (i,j,l) in localflip]
            flipfactor = np.prod(flipfactors)
            if flipfactor == -1:
                print("(i,j,l) = (",i,",",j,",",l,")")
        
        if version == 0:
            if rot == 0:
                k = (i + 2*j +shift)
                m = (i - j+shift)
                if k%6 == 0:
                    if m%6 == 0:
                        if l == 1:
                            spinstate[s] = sign
                        else:
                            spinstate[s] = - sign
                    if m%6 == 3:
                        if l == 0:
                            spinstate[s] = sign
                        else:
                            spinstate[s] = - sign

                if k%6 == 1:
                    if m%6 == 1:
                        if l== 2:
                            spinstate[s] = - sign
                        else:
                            spinstate[s] = sign
                    if m%6 == 4:
                        if l == 1:
                            spinstate[s] = - sign
                        else:
                            spinstate[s] = sign
                if k%6 == 2:
                    if m%6 == 2:
                        if l == 1:
                            spinstate[s] = sign
                        else:
                            spinstate[s] = - sign
                    if m%6 == 5:
                        if l == 0:
                            spinstate[s]= flipfactor*sign
                        elif l == 1:
                            spinstate[s] = - sign
                        elif l == 2:
                            spinstate[s] = -flipfactor*sign
                if k%6 == 3:
                    if m % 6 == 0:
                        if l == 2:
                            spinstate[s] = sign
                        elif l == 0:
                            spinstate[s] = - sign
                        elif l == 1:
                            spinstate[s] = - flipfactor*sign
                    if m % 6 == 3:
                        if l == 1:
                            spinstate[s] = flipfactor*sign
                        else:
                            spinstate[s] = sign

                if k%6 == 4:
                    if m%6 == 1:
                        if l == 2:
                            spinstate[s] = sign
                        else:
                            spinstate[s] = - sign
                    if m%6 == 4:
                        if l == 1:
                            spinstate[s] = sign
                        else:
                            spinstate[s] = - sign

                if k%6 == 5:
                    if m%6 == 2:
                        if l == 0:
                            spinstate[s] = - flipfactor*sign
                        else:
                            spinstate[s] = sign
                    if m%6 == 5:
                        if l == 1:
                            spinstate[s] = - sign
                        elif l == 0:
                            spinstate[s] = sign
                        elif l == 2:
                            spinstate[s] = flipfactor*sign
    
    spinstate = spinstate.astype('int8')


# In[ ]:

def IntermediateInit(spinstates, nt, nh, s_ijl, same):
    if same:
        spinstate = np.zeros(len(s_ijl), dtype = 'int8')
        version = 0
        sign = np.random.randint(0,2, dtype = 'int8')*2-1
        shift = np.random.randint(0,6)
        rot = 0
        IntermediateInit(spinstate, s_ijl,version, sign, shift, rot)
        
        for t in range(nt*nh):
            spinstates[t] = np.copy(spinstate.astype('int8'))
    else:
        versions = [0 for w in range(nt*nh)]
        signs = [np.random.randint(0,2, dtype = 'int8')*2-1 for w in range(nt*nh)]
        shifts = [np.random.randint(0,6) for w in range(nt*nh)]
        rots = [0 for w in range(nt*nh)]
        for t in range(nt*nh):
            IntermediateInitOneState(spinstates[t], s_ijl,versions[t], signs[t], shifts[t], rots[t])


# In[ ]:

def LargeJ3InitOneState(spinstate, s_ijl, version, sign, shift, rot):
    if version == 0:
        # for now only one version
        if rot == 0:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (2*i + j + shift)%6
                if loc == 0:
                    if l == 2:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = - sign
                if loc == 1:
                    if l == 2:
                        spinstate[s] = - sign
                    else:
                        spinstate[s] = sign
                if loc == 2:
                    if l == 2:
                        spinstate[s] = sign
                    elif l == 1:
                        spinstate[s] = - sign
                    elif l == 0:
                        spinstate[s] = np.random.randint(0, 2)*2 -1
                if loc == 3:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = - sign
                if loc == 4:
                    if l == 1:
                        spinstate[s] = - sign
                    else:
                        spinstate[s] = sign
                if loc == 5:
                    if l == 1:
                        spinstate[s] = sign
                    elif l == 2:
                        spinstate[s] = - sign
                    elif l == 0:
                        spinstate[s] = np.random.randint(0, 2)*2 -1
        if rot == 1:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i + 2*j + shift)%6
                if loc == 0:
                    spinstate[s] = sign
                if loc == 1:
                    if l == 1:
                        spinstate[s] = np.random.randint(0, 2)*2 -1
                    else:
                        spinstate[s] = -sign
                if loc == 2:
                    if l == 1:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if loc == 3:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if loc == 4:
                    if l == 1:
                        spinstate[s] = np.random.randint(0, 2)*2 -1
                    else:
                        spinstate[s] = sign
                if loc == 5:
                    spinstate[s] = -sign
        if rot == 2:
            for s, (i, j, l) in enumerate(s_ijl):
                loc = (i - j + shift)%6
                if loc == 0:
                    if l == 0:
                        spinstate[s] = -sign
                    elif l == 1:
                        spinstate[s] = sign
                    elif l == 2:
                        spinstate[s] = np.random.randint(0, 2)*2 -1
                if loc == 1:
                    if l == 0:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign  
                if loc == 2:
                    if l == 0:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if loc == 3:
                    if l == 0:
                        spinstate[s] = sign
                    elif l == 1:
                        spinstate[s] = -sign
                    elif l == 2:
                        spinstate[s] = np.random.randint(0, 2)*2 -1
                if loc == 4:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if loc == 5:
                    if l == 1:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
    if version == 1:
        print("Warning: LargeJ3 test version!!!")
        # for now only one version
        for s, (i, j, l) in enumerate(s_ijl):
            loc = (i + 2*j + shift)%3
            if loc == 0:
                if l == 0:
                    spinstate[s] = np.random.randint(0, 2)*2 -1
                else:
                    spinstate[s] = sign
            if loc == 1:
                spinstate[s] = - sign
            if loc == 2:
                if l == 1:
                    spinstate[s] = -sign
                else:
                    spinstate[s] = sign
    spinstate = spinstate.astype('int8')         


# In[ ]:

def LargeJ3Init(spinstates, nt, nh, s_ijl, same):
    if same:
        version = 0
        sign = np.random.randint(0,2, dtype = 'int8')*2-1
        shift = np.random.randint(0,6)
        rot = np.random.randint(0,3)
        
        spinstate = np.zeros(len(s_ijl), dtype = 'int8')
        LargeJ3InitOneState(spinstate, s_ijl, version, sign, shift, rot)
        spinstates = [np.copy(spinstate.astype('int8')) for w in range(nt*nh)]
    else:
        versions = [0 for w in range(nt*nh)]
        signs = [np.random.randint(0,2, dtype = 'int8')*2-1 for w in range(nt*nh)]
        shifts = [np.random.randint(0,6) for w in range(nt*nh)]
        rots = [np.random.randint(0,3) for w in range(nt*nh)]
        
        for t in range(nt*nh):
            LargeJ3InitOneState(spinstates[t], s_ijl, versions[t], signs[t], shifts[t], rots[t])


# In[ ]:

def DipolarToJ4Init(spinstates, nt, s_ijl):
    for t in range(nt*nh):
        sign = np.random.randint(0,2, dtype = 'int8')*2-1
        for s, (i, j, l) in enumerate(s_ijl):
            jp = i + 2*j
            ######### i = 0 ###############
            if i%4 == 0:
                if jp%12 == 0:
                    if l == 0:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign

                if jp%12 == 2:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if jp%12 == 4:
                    if l == 1:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 6:
                    if l == 2:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 8:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if jp%12 == 10:
                    if l == 1:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign

            ########## i = 1 ##############
            if i%4 == 1:
                if jp%12 == 1:
                    if l == 1:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 3:
                    if l == 2:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if jp%12 == 5:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if jp%12 == 7:
                    if l == 2:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 9:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if jp%12 == 11:
                    if l == 2:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign


            ########## i = 2 ##############
            if i%4 == 2:
                if jp%12 == 0:
                    spinstate[s] = -sign
                if jp%12 == 2:
                    if l == 0:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 4:
                    if l == 0:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 6:
                    spinstate[s] = -sign
                if jp%12 == 8:
                    if l == 2:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 10:
                    if l == 2:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign


            ########## i = 3 ##############
            if i%4 == 3:
                if jp%12 == 1:
                    if l == 0:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 3:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if jp%12 == 5:
                    if l == 0:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 7:
                    if l == 1:
                        spinstate[s] = -sign
                    else:
                        spinstate[s] = sign
                if jp%12 == 9:
                    if l == 0:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
                if jp%12 == 11:
                    if l == 1:
                        spinstate[s] = sign
                    else:
                        spinstate[s] = -sign
    spinstate = spinstate.astype('int8')


# In[ ]:

def J1Init(spinstates, nt, nh, s_ijl, same):
    #print('In J1 init')
    if same:
        #print('In J1 init same')
        version = np.random.randint(0,2)
        if version == 0:
            J1J2Init(spinstates, nt, s_ijl, same)
        else:
            J1J3Init(spinstates, nt, s_ijl, same)
    else:
        #print('In J1 init diff')
        for t in range(nt*nh):
            for h in range(nh):
                w = t*nh+h        
                version = np.random.randint(0,2)
                if version == 0:
                #    print("version = J1J2")
                    J1J2Init([spinstates[w]],1,1, s_ijl, same)
                else:
                #    print("version = J1J3")
                    J1J3Init([spinstates[w]], 1,1, s_ijl, same)


# In[ ]:

def determine_init(hamiltonian, magninit, **kwargs):
    '''
        Returns the type of init
    '''
    # Extract the hamiltonian
    if hamiltonian:
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

        inittype = ""
        if J1!=0:
            inittype+="J1"
        if J2!=0:
            inittype+="J2"
        if J3!=0:
            inittype+="J3"
        if J4!=0:
            inittype+="J4"

        if inittype == "J1J2J3":
            if J3/J2 <= 0.5 and J3 > 0:
                inittype += "LJ2"
            if J3/J2 > 0.5 and J3/J2 <= 1:
                inittype += "Intermediate"
            if J3/J2 > 1:
                inittype += "LJ3"

    if magninit:
        maxflip = kwargs.get('maxflip', False)
        magnstripes = kwargs.get('magnstripes', False)
        
        if not maxflip and not magnstripes:
            inittype = "magninit"
        elif maxflip:
            inittype = "maxflip"
        elif magnstripes:
            inittype = "magnstripes"
    return inittype
    


# In[ ]:

def statesinit(nt, nh, hfields, id2walker, d_ijl, d_2s, s_ijl,
               hamiltonian, random = True, same = False,\
               magninit = False, **kwargs):
    '''
       This function associates a dimer state to each dual bond. By default, this is done by first associating randomly a spin to
       each spin site and then map this onto a dimer state. If the initial state isn't random, it is what is believed to be the ground state
       of the problem.
       It returns the list of the states for each temperature and the energies.
    '''


    #initialize the dimers
    states = [np.array([1 for i in range(len(d_ijl))], dtype='int8') for ignored in range(nt*nh)]
    #initialize the spins randomly
    spinstates = [(np.random.randint(0, 2, size=len(s_ijl), dtype = 'int8')*2 - 1) for i in range(nt*nh)]
    if not random:
        inittype = determine_init(hamiltonian, magninit, **kwargs) 
        print("Initialisation type: ", inittype)
        if inittype == "magninit":
            magnetisedInit(spinstates, nt, nh, s_ijl, hfields, same)
        elif inittype == "maxflip":
            magnetisedInitMaxFlip(spinstates, nt, nh, s_ijl, hfields, same)
        elif inittype == "magnstripes":
            magnetisedInitStripes(spinstates, nt, nh, s_ijl, hfields, same)
        elif inittype == "J1":
            print('J1Init')
            J1Init(spinstates, nt, nh, s_ijl, same)
        elif inittype == "J1J2":
            J1J2Init(spinstates, nt, nh, s_ijl, same)
        elif inittype == "J1J3":
            J1J3Init(spinstates, nt, nh, s_ijl, same)
        elif inittype == "J1J2J3LJ2":
            LargeJ2Init(spinstates, nt, nh, s_ijl, same)
        elif inittype == "J1J2J3Intermediate":
            IntermediateInit(spinstates, nt, nh, s_ijl, same)
        elif inittype == "J1J2J3LJ3":
            LargeJ3Init(spinstates, nt,nh, s_ijl, same)
        elif inittype == "J1J2J3J4":
            DipolarToJ4Init(spinstates, nt*nh, s_ijl, same)
        else:
            print("Something went wrong and you get something random instead")
    states = np.array(states, 'int32')
    spinstates = np.array(spinstates, 'int32')
    ##initialise the dimer state according to the spin state
    for bid in range(nt):
        for hid in range(nh): 
            i = id2walker[bid, hid]
            for id_dim in range(len(d_ijl)):
                [id_s1, id_s2] = d_2s[id_dim]
                s1 = spinstates[i][id_s1]
                s2 = spinstates[i][id_s2]
                if (s1 == s2):
                    states[i][id_dim] = 1
                else:
                    states[i][id_dim] = -1
                    
            states[i] = states[i].astype('int8')
            
    # compute the energy of the initialized states (via the function in c++)
    en_states = [[compute_energy(hamiltonian, states[id2walker[bid, hid]])
                  - hfields[hid]*spinstates[id2walker[bid,hid]].sum()
                  for hid in range(nh)]
                 for bid in range(nt)]
    
    en_states = np.array(en_states)
    #
    return states, en_states, spinstates

