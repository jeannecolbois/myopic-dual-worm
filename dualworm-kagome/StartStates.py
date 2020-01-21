
# coding: utf-8

# In[1]:


from DualwormFunctions import compute_energy
import numpy as np


# In[2]:


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
        
        for t in range(nt):
            for h in range(nh):
                w = t*nh+h
                newspinstates = [(np.random.randint(0, 2, size=len(s_ijl))*2 - 1)]
                if versions[w]:
                    magnetisedInitStripes(newspinstates, 1, 1, s_ijl, [hfields[h]], True)
                else:
                    magnetisedInitMaxFlip(newspinstates, 1, 1, s_ijl, [hfields[h]], True)
                
                spinstates[w] = newspinstates[0]


# In[ ]:


def magnetisedInitStripes(spinstates, nt, nh, s_ijl, hfields, same):
   
    if same:
        
        if hfields[0] == 0:
            sign = np.random.randint(0,2)*2-1
        else:
            sign = np.sign(hfields[0])
        
        version = np.random.randint(0,3)
        
        spinstate = sign*np.ones(len(s_ijl))
        
        for s, (i,j,l) in enumerate(s_ijl):
            if l == version:
                spinstate[s] = -sign
                
        for w in range(nt*nh):
            spinstates[w] = np.copy(spinstate) # to make sure that it is DIFFERENT states
        
    else:
        print("Magnetised init - stripes")
        versions = [np.random.randint(0,3) for w in range(nt*nh)]
        
        signs = [np.sign(h) for t in range(nt) for h in hfields]
        
        for w in range(nt*nh):
            if signs[w] == 0:
                signs[w] = np.random.randint(0,2)*2-1
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
        
        signs = [np.sign(h) for t in range(nt) for h in hfields]
        
        for w in range(nt*nh):
            if signs[w] == 0:
                signs[w] = np.random.randint(0,2)*2-1
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
                    
                        


# In[3]:


def J1J2Init(spinstates, nt, s_ijl, same):
    if same:
        #print("In same (J1J2)")
        version = np.random.randint(0,3)
        sign = np.random.randint(0,2)*2 -1
        
        spinstate = np.ones(len(s_ijl))
        for s, (i,j,l) in enumerate(s_ijl):
            if l == version:
                spinstate[s] = sign
            if l == (version+1)%3:
                spinstate[s] = -sign
            if l == (version+2)%3:
                spinstate[s] = np.random.randint(0,2)*2-1
        
        for t in range(nt):
            spinstates[t] = np.copy(spinstate)
    else:
        #print("In Differents (J1J2)")
        versions = [np.random.randint(0,3) for t in range(nt)]
        signs = [np.random.randint(0,2)*2 -1 for t in range(nt)]
        #print("versions and signs generated (J1J2)")
        for t in range(nt):
            for s, (i, j, l) in enumerate(s_ijl):
                if l == versions[t]:
                    spinstates[t][s] = signs[t]
                if l == (versions[t] + 1)%3:
                    spinstates[t][s] = -signs[t]
                if l == (versions[t] + 2)%3:
                    spinstates[t][s] = np.random.randint(0,2)*2 - 1


# In[4]:


def J1J3Init(spinstates, nt, s_ijl, same):

        
    if same:
        sign = np.random.randint(0,2)*2 -1
        
        spinstate = np.ones(len(s_ijl))
        for s, (i,j,l) in enumerate(s_ijl):
            subl = 3*((i + 2*j)%3) + l
            if subl in (0, 2, 4):
                spinstate[s] = sign
            if subl in (3, 5, 7):
                spinstate[s] = - sign
            if subl in (1, 6, 8):
                spinstate[s] = np.random.randint(0,2)*2 -1          
            
        for t in range(nt):
            spinstates[t] = np.copy(spinstate)
    else:
        signs = [np.random.randint(0,2)*2 -1 for t in range(nt)]
    
        for t in range(nt):
            for s, (i,j,l) in enumerate(s_ijl):
                subl = 3*((i + 2*j)%3) + l
                if subl in (0, 2, 4):
                    spinstates[t][s] = signs[t]
                if subl in (3, 5, 7):
                    spinstates[t][s] = - signs[t]
                if subl in (1, 6, 8):
                    spinstates[t][s] = np.random.randint(0,2)*2 -1


# In[5]:


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


# In[6]:


def LargeJ2Init(spinstates, nt, s_ijl, same):
    if same:
        version = np.random.randint(0,3)
        sign = np.random.randint(0,2)*2-1
        shift = np.random.randint(0,4)
        rot = np.random.randint(0,3)
        
        spinstate = np.zeros(len(s_ijl))
        LargeJ2InitOneState(spinstate, s_ijl, version, sign, shift, rot)
        spinstates = [np.copy(spinstate) for t in range(nt)]
    else:
        versions = [np.random.randint(0,3) for t in range(nt)]
        signs = [np.random.randint(0,2)*2-1 for t in range(nt)]
        shifts = [np.random.randint(0,4) for t in range(nt)]
        rots = [np.random.randint(0,3) for t in range(nt)]
        
        for t in range(nt):
            LargeJ2InitOneState(spinstates[t], s_ijl, versions[t], signs[t], shifts[t], rots[t])


# In[7]:


def IntermediateInitOneState(spinstate, s_ijl):
    sign = np.random.randint(0,2)*2-1
    for s, (i, j, l) in enumerate(s_ijl):
        k = i + 2*j
        m = i - j
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
                    spinstate[s]= sign
                else:
                    spinstate[s] = - sign
        if k%6 == 3:
            if m % 6 == 0:
                if l == 2:
                    spinstate[s] = sign
                else:
                    spinstate[s] = - sign
            if m % 6 == 3:
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
                    spinstate[s] = - sign
                else:
                    spinstate[s] = sign
            if m%6 == 5:
                if l == 1:
                    spinstate[s] = - sign
                else:
                    spinstate[s] = sign


# In[8]:


def IntermediateInit(spinstates, nt, s_ijl, same):
    if same:
        spinstate = np.zeros(len(s_ijl))
        IntermediateInit(onestate, s_ijl)
        
        for t in range(nt):
            spinstates[t] = np.copy(spinstate)
    else:
        for t in range(nt):
            IntermediateInitOneState(spinstates[t], s_ijl)


# In[9]:


def LargeJ3Init(spinstates, nt, s_ijl, same):
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


# In[10]:


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


# In[11]:


def J1Init(spinstates, nt, s_ijl, same):
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
        for t in range(nt):
            version = np.random.randint(0,2)
            if version == 0:
            #    print("version = J1J2")
                J1J2Init([spinstates[t]],1, s_ijl, same)
            else:
            #    print("version = J1J3")
                J1J3Init([spinstates[t]], 1, s_ijl, same)


# In[12]:


def determine_init(hamiltonian, magninit, **kwargs):
    '''
        Returns the type of init
    '''
    # Extract the hamiltonian
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
    


# In[13]:


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
    states = [np.array([1 for i in range(len(d_ijl))], dtype='int32') for ignored in range(nt*nh)]
    #initialize the spins randomly
    spinstates = [(np.random.randint(0, 2, size=len(s_ijl))*2 - 1) for i in range(nt*nh)]
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
            J1Init(spinstates, nt*nh, s_ijl, same)
        elif inittype == "J1J2":
            J1J2Init(spinstates, nt*nh, s_ijl, same)
        elif inittype == "J1J3":
            J1J3Init(spinstates, nt*nh, s_ijl, same)
        elif inittype == "J1J2J3LJ2":
            LargeJ2Init(spinstates, nt*nh, s_ijl, same)
        elif inittype == "J1J2J3intermediate":
            intermediateInit(spinstates, nt*nh, s_ijl, same)
        elif inittype == "J1J2J3LJ3":
            LargeJ3Init(spinstates, nt*nh, s_ijl, same)
        elif inittype == "J1J2J3J4":
            DipolarToJ4Init(spinstates, nt*nh, s_ijl, same)
            
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
                    
    # compute the energy of the initialized states (via the function in c++)
    en_states = [[compute_energy(hamiltonian, states[id2walker[bid, hid]])
                  - hfields[hid]*spinstates[id2walker[bid,hid]].sum()
                  for hid in range(nh)]
                 for bid in range(nt)]
    
    en_states = np.array(en_states)
    #
    return states, en_states, spinstates


# In[15]:


#def state7shaped(s_ijl):
#    spinstate7 = np.random.randint(0, 2, size=len(s_ijl))*2 - 1
#
#    for s, (i, j, l) in enumerate(s_ijl):
#        if (2*i + j)%4 == 0:
#            if l == 0:
#                spinstate7[s] = 1
#            if l == 1:
#                spinstate7[s] = -1
#            if l == 2:
#                spinstate7[s] = -1
#        elif (2*i + j)%4 == 1:
#            if l == 0:
#                spinstate7[s] = 1
#            if l == 1:
#                spinstate7[s] = -1
#            if l == 2:
#                spinstate7[s] = 1
#        elif (2*i + j)%4 == 2:
#            if l == 0:
#                spinstate7[s] = 1
#            if l == 1:
#                spinstate7[s] = 1
#            if l == 2:
#                spinstate7[s] = -1
#        elif (2*i + j)%4 == 3:
#            if l == 0:
#                spinstate7[s] = 1
#            if l == 1:
#                spinstate7[s] = -1
#            if l == 2:
#                spinstate7[s] = -1
#
#    return spinstate7

