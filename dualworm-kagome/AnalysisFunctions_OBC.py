
# coding: utf-8

# Last update 19.12.2019
# 
# Author : Jeanne Colbois
# 
# Please send any comments, questions or remarks to Jeanne Colbois: jeanne.colbois@epfl.ch.
# The author would appreciate to be cited in uses of this code, and would be very happy to hear about potential nice developments.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import KagomeFunctions_OBC as kf # "library" allowing to work on Kagome
import KagomeDrawing as kdraw
import itertools


# In[ ]:


def ComputeNthCorrList(NNListN, sconf, x, y, factor):
    '''
        Compute the correlations for the Nth neighbours pairs given in NNListN,
        where sconf is the list of spin configurations (sconf[s,i] is the value of
        the spin at index s in the ith configuration) and x and y give the position of s
    '''


    nNth = len(NNListN)

    # For now, we assume that the various correlations are uncorrelated, and we work with l = 1

    l = sconf[0].size
    assert l == 1

    s1 = 0
    s2 = 0

    ldist = []

    for pair in NNListN:
        s1 += sconf[pair[0]]/nNth# computing the average s1 over configurations
        s2 += sconf[pair[1]]/nNth # computing the average s2 over configurations

        distval = expdist(pair[0], pair[1], x,y, factor)
        ldist.append(distval)

    w12 = np.array([(sconf[pair[0]]-s1)*(sconf[pair[1]]-s2) for pair in NNListN]);
    avgw12 = sum(w12)/nNth
    cov12 = avgw12*nNth/(nNth-1)
    varw12 = sum((w12-avgw12)**2)/(nNth-1)
    varcov12 = varw12*nNth/((nNth-1)**2)

    ldist = np.array(ldist)
    avgdist = sum(ldist)/nNth
    vardist = sum((ldist - avgdist)**2)/(nNth-1)


    return cov12, varcov12, avgdist, vardist


# In[ ]:


def ComputeNthCorr(firstpairs, rss, s42_realpos):
    '''
        Compute the correlations for the Nth neighbours pairs given in firstpairs, where rss is the
        list of spin configurations (for each spin s, r[s] is the list of values this spin takes)
        and s42_realpos gives the position of each spin in pixels.
    '''
    nthcorr = []
    nthvar = []
    acccorr = 0
    accvar = 0
    nthdist = []
    nthavgdist = 0

    for pair in firstpairs:
        l = rss[pair[0]].size

        ## Correlations
        s1 = sum(rss[pair[0]])/l
        s2 = sum(rss[pair[1]])/l
        w12 = np.array([(rss[pair[0]][i]-s1)*(rss[pair[1]][i]-s2) for i in range(l)]) ## Local variable measuring the correlation between spin pairs
        avgw12 = sum(w12)/l
        cov12 = avgw12*l/(l-1) ## Estimator of the correlation between spin pair[0] and pair[1]
        varw12 = sum((w12-avgw12)**2)/(l-1)
        varcov12 = varw12*l/((l-1)**2)
        nthcorr.append(cov12)
        nthvar.append(varcov12)

        ## Accumulate correlations over given neighbour type
        acccorr += cov12/firstpairs[:,0].size
        accvar += varcov12/firstpairs[:,0].size

        ## True distance (pixels)
        truedist = np.linalg.norm(s42_realpos[pair[0]]- s42_realpos[pair[1]])
        nthdist.append(truedist)
        nthavgdist += truedist/firstpairs[:,0].size

    ## Variance on true distance estimator (pixels)
    nthvardist = 0
    for truedist in nthdist:
        nthvardist += ((truedist-nthavgdist)**2)/(len(nthdist)-1)


    return nthcorr, acccorr, nthvar, accvar, nthdist, nthavgdist, nthvardist


# In[ ]:


def LoadSpinConfigsLarge(foldername, spinconfigfile, imgfile, alpha = 0.3, factor = 1):
    configlist = np.loadtxt(foldername+spinconfigfile,delimiter=',')
    x = configlist[:,0]
    y = configlist[:,1]
    sconf = configlist[:,2:]

    spos = [(x[i],y[i]) for i in range(len(x))]

    lup =[]
    ldown = []
    for ids, sval in enumerate(sconf[:,0]):
        if sval == 1:
            lup.append(ids)
        else:
            ldown.append(ids)

    fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    if imgfile == "":
        plotSpinSites(foldername, imgfile, x/factor, y/factor, lup, putimage = False, color = 'green', alpha = alpha)
    else:
        plotSpinSites(foldername, imgfile, x/factor, y/factor, lup, putimage = True, color = 'green', alpha = alpha)
    plotSpinSites(foldername, imgfile, x/factor, y/factor, ldown, putimage = False, color = 'orange', alpha = alpha)
    #spinsimg = mpimg.imread(foldername+imgfile)
    #fig, ax = plt.subplots(figsize = (6,6),dpi=200)
    #plt.imshow(spinsimg)
    #plt.plot(xdown,ydown,'.', alpha = alpha, color = 'orange')
    #plt.plot(xup, yup, '.', alpha = alpha, color = 'green')
    #plt.axis('equal');

    return configlist, x, y, sconf


# In[ ]:


def DrawClusterOnConfig(secondfoldername, L, refid, spos, factor = 1, doplot = False,
                        domap = False, a = 2, **kwargs):
    '''
        Creates a cluster of side size L (9L^2 sites tot)
        Maps it onto the configuration with 
        refloc =  [(L-1, 0, 0), (L-1, 0, 1), (L-1, 0, 2), (2L-1, 0)]
        in the usual language
    '''
    
    ## Getting the reference positions in the lattice and in the experimental lattice
    # exp lattice
    numdots = len(refid)
    refpos = np.zeros((numdots,2))
    for i in range(numdots):
        refpos[i,:] = spos[refid[i],:]
    print(refpos)
    plotSpinSites(secondfoldername, "", refpos[:,0], refpos[:,1],[i for i in range(numdots)], 
                  putimage = False, color = "purple", alpha = 1)
    plotSpinSites(secondfoldername, "", np.array([refpos[numdots-1,0]]),
                  np.array([refpos[numdots-1,1]]),[0], putimage = False, color = "magenta", alpha = 1)
    plotSpinSites(secondfoldername, "", np.array([refpos[0,0]]), np.array([refpos[0,1]]),[0],
                  putimage = False, color = "purple", alpha = 1)    
    
    # actual lattice
    s_ijl, ijl_s = kf.createspinsitetable(L)
    sv_ijl, ijl_sv, e_2sv, pos = kf.graphkag(L,2)
    ori = np.array([pos[ijl_sv[(L-1, 0, 0)]], pos[ijl_sv[(L-1, 0, 1)]],pos[ijl_sv[(L-1, 0, 2)]],
                    pos[ijl_sv[(2*L-1, 0, 0)]]])
    pos = np.array([pos[i] for i in range(len(pos))])
    
    # 0) Checking the factor
    dist12_ori = np.linalg.norm(ori[0] - ori[3])
    dist12_map = np.linalg.norm(refpos[0] - refpos[3])
    
    if not abs(dist12_ori - dist12_map) < 1e-8:
        print("Help for factor: currently {0} and {1}".format(dist12_ori, dist12_map))

    # 1) Plotting the situation (to test before doing map)
    plotSpinSites(secondfoldername, "", np.array([ori[0,0]]), np.array([ori[0,1]]),[0],
                  putimage = False, color = "red", alpha = 1)
    plotSpinSites(secondfoldername, "", np.array([ori[1,0]]), np.array([ori[1,1]]),[0],
                  putimage = False, color = "pink", alpha = 1)
    plotSpinSites(secondfoldername, "", np.array([ori[2,0]]), np.array([ori[2,1]]),[0],
                  putimage = False, color = "purple", alpha = 1)    
    plotSpinSites(secondfoldername, "", np.array([ori[3,0]]), np.array([ori[3,1]]),[0],
                  putimage = False, color = "pink", alpha = 1)    
    plotSpinSites(secondfoldername, "", pos[:,0], pos[:,1], [i for i in range(len(pos))],
                  putimage = False, color = "blue", alpha = 0.1)
    
    # 2) Performing the map
    if domap:
        
        fig, ax = plt.subplots(figsize = (8,8),dpi=200)
        # We just want one solution --- 
        # 1) check crossing of lines 1-2:
        # -- a) Find the angle 
        nref = np.dot(refpos[3]-refpos[0], refpos[3]-refpos[0])
        nori = np.dot(ori[3]-ori[0], ori[3] - ori[0])
        cosangle = np.dot(refpos[3]-refpos[0],ori[3]-ori[0])/np.sqrt(nref*nori)
        sinangle = np.sqrt(1-cosangle**2)
        rot = np.array([[cosangle, sinangle],[-sinangle, cosangle]])
        
        rotrefpos = np.dot(rot, refpos.T).T
        rotpos = np.dot(rot, spos.T).T
        #      (plot to check)
        plotSpinSites(secondfoldername, "", rotrefpos[:,0], rotrefpos[:,1],
                      [i for i in range(len(rotrefpos))], putimage = False, color = "red", alpha = 0.5)
        plotSpinSites(secondfoldername, "", pos[:,0], pos[:,1],
                      [i for i in range(len(pos))], putimage = False, color = "blue")
        # -- b) Find the translation 
        fig, ax = plt.subplots(figsize = (8,8),dpi=200)
        tr = ori[0]-rotrefpos[0] 
        mappos = rotpos.T
        mappos[0,:] += tr[0]
        mappos[1,:] += tr[1]
        mappos = mappos.T
        #      (plot to check)
        plotSpinSites(secondfoldername, "", mappos[:,0], mappos[:,1],
                      [i for i in range(len(mappos))], putimage = False, **kwargs)
        plotSpinSites(secondfoldername, "", pos[:,0], pos[:,1],
                      [i for i in range(len(pos))], putimage = False, color = "blue")
        
        # -- c) express the (x,y) positions in (m1, m2) coordinates (i.e. on the [a1,a2] basis)
        m1m2 = mappos
        m1m2[:,0] = (1/a) * (mappos[:,0] - mappos[:,1]/np.sqrt(3))
        m1m2[:,1] = (2/a) * (mappos[:,1]/np.sqrt(3))
        
        # -- d) now, from m1m2, get (i,j,l)
        m1m2[:,0] += -1/2 # translate from -1/2 a1
        # check if l == 0:
        ijl = np.zeros((len(mappos),3), dtype = 'int')
        for s in range(len(mappos)):
            if abs(m1m2[s,0] - round(m1m2[s,0])) > 1e-2:
                # then l = 1
                i = m1m2[s,0] + 0.5
                j = m1m2[s,1] - 0.5
                if abs(i - round(i)) < 1e-2 and abs(j - round(j)) < 1e-2:
                    ijl[s,:] = np.array([round(i),round(j),1])
            else:
                # then l = 0 or 2
                if abs(m1m2[s,1] - round(m1m2[s,1])) > 1e-2:
                    # then l = 2
                    i = m1m2[s,0] + 1
                    j = m1m2[s,1] - 0.5
                    if abs(i - round(i)) < 1e-2 and abs(j - round(j)) < 1e-2:
                        ijl[s,:] = np.array([round(i),round(j),2])
                else:
                    # then l = 0
                    i = m1m2[s,0]
                    j = m1m2[s,1]
                    if abs(i - round(i)) < 1e-2 and abs(j - round(j)) < 1e-2:
                        ijl[s,:] = np.array([round(i),round(j),0])
    else:
        rot = np.array([[1,0],[0,1]])
        tr = np.array([0,0])
        m1m2 = np.array([])
        ijl = np.array([])
    # 2) if too far check crossing of lines 2-3
    # 3) If both are basically a zero angle assume that it is only a translation
    #   and just find the difference

    #if not doplot:
    #    print("Original: ", ori)
    #else:
    #    if doplot and x.size and y.size and sconf.size: # testing if not empty
    #        fig, ax = plt.subplots(figsize = (8,8),dpi=200)
    #        plotSpinSites(foldername, "", x/factor, y/factor, sconf[:,0],
    #               putimage = False, color = 'lightblue', alpha = alpha)
    
    return rot,tr,m1m2,ijl


# In[1]:


def StrctFact(ijl,m1m2, sconf,L, factor = 1):
    (q_k1k2, k1k2_q) = kdraw.KagomeReciprocal(L)
    q_k1k2 = np.array(q_k1k2)
    
    StrctFact = np.zeros((q_k1k2.shape[0],3, 3), dtype = 'complex128')
    nspins = len(sconf)
    m = sum(sconf)/sum(abs(sconf))
    N = np.sqrt((nspins**2)/2)
    for s1 in range(nspins):
        (i1,j1,l1) = ijl[s1]
        for s2 in range(s1+1, nspins):
            # correlation
            c = np.asscalar(sconf[s1]*sconf[s2]-m**2)
            # structure factor computation
            (i2,j2,l2) = ijl[s2]
            exponent = 1j*2 * np.pi * np.dot(q_k1k2, (m1m2[s1]-m1m2[s2]))
            StrctFact[:,l1,l2] += c * np.exp(exponent)/N
            StrctFact[:,l2,l1] += c * np.exp(-exponent)/N
            
    return StrctFact, m


# In[ ]:


def plotSpinSites(foldername, imgfile, x, y, listsites, putimage = True, marker = '.', color = 'blue', alpha = 0.3, linestyle='none'):
    xplot = x[listsites]
    yplot = y[listsites]
    if putimage == True:
        spinsimg = mpimg.imread(foldername+imgfile)
        fig, ax = plt.subplots(figsize = (12,12),dpi=300)
        plt.imshow(spinsimg)
    plt.plot(xplot,yplot,marker = marker, linestyle = linestyle, alpha = alpha, color = color)
    plt.axis('equal')


# In[ ]:


def plotSpinPairs(foldername, imgfile, x, y, listpairs, putimage = True, marker = '.', color = 'blue', alpha = 0.3):
    plotSpinSites(foldername, imgfile, x,y, list(listpairs[0]), putimage = putimage, marker = marker,color = color, alpha = alpha, linestyle = 'solid')
    for pair in listpairs:
        plotSpinSites(foldername, imgfile, x, y, list(pair), putimage = False, marker = marker, color = color, alpha= alpha, linestyle = 'solid')


# In[ ]:


def expdist(sid1, sid2, x, y, factor):
    return np.sqrt((x[sid1]-x[sid2])**2 +(y[sid1]-y[sid2])**2)/factor


# In[ ]:


def KagomeLatticeHistogram(x, y, factor = 20):
    nspins = len(x)
    distances = []
    distances_s1s2 = {}
    for sid1 in range(nspins):
        for sid2 in range(sid1+1, nspins):
            distval = expdist(sid1,sid2, x,y, factor)
            distances.append(distval)
            if distval in distances_s1s2:
                distances_s1s2[distval].append((sid1,sid2))
            else:
                distances_s1s2[distval] = [(sid1,sid2)]


    distances = np.array(distances)
    fig, ax = plt.subplots(dpi=200)
    plt.hist(distances, bins = [0.95, 1.05, 1.68, 1.78, 1.95, 2.05, 2.59, 2.69, 2.95, 3.05, 3.41, 3.52, 3.54, 3.67, 3.95, 4.05, 4.1])
    plt.show()

    return distances_s1s2


# In[ ]:


def KagomeLatticeNeighboursLists(distances_s1s2, distconds):
    nn = len(distconds)
    NNList = [[] for i in range(nn)]
    for k in distances_s1s2.keys():
        for i in range(nn):
            if distconds[i][0] < k <distconds[i][1]:
                for val in distances_s1s2[k]:
                    NNList[i].append(val)

    # correcting the lists with different types of neighbours at the same distance
    NNList3_0 = []
    NNList3_1 = []
    for (s1,s2) in NNList[2]:
        s1nn = [pair[1 - pair.index(s1)] for pair in NNList[0] if s1 in pair]
        s2nn = [pair[1 - pair.index(s2)] for pair in NNList[0] if s2 in pair]
        intersect = list(set(s1nn) & set(s2nn))

        if len(intersect) == 0:
            NNList3_1.append((s1,s2))
        else:
            NNList3_0.append((s1,s2))

    NNList6_0 = []
    NNList6_1 = []
    for (s1,s2) in NNList[5]:
        s1nn = [pair[1 - pair.index(s1)] for pair in NNList[1] if s1 in pair]
        s2nn = [pair[1 - pair.index(s2)] for pair in NNList[1] if s2 in pair]
        intersect = list(set(s1nn) & set(s2nn))

        if len(intersect) == 0:
            NNList6_1.append((s1,s2))
        else:
            NNList6_0.append((s1,s2))

    # replacing in NNList
    NNList[2] = []
    NNList[2] = NNList3_0
    NNList.insert(3, NNList3_1)
    NNList[6] = NNList6_0
    NNList.insert(7, NNList6_1)
    return NNList


# In[ ]:


def KagomeLatticeTriangles(NNList,sizelatt):
    trianglelist = []

    for s1 in range(sizelatt):
        s1tuples = [(sa, sb) for (sa, sb) in NNList[0] if sa == s1]
        if len(s1tuples) == 2:
            s2 = s1tuples[0][1]
            s3 = s1tuples[1][1]
            if (min(s2,s3), max(s2,s3)) in NNList[0]:
                triangle = [(s1,s2),(s2,s3),(s1,s3)]
                if triangle not in trianglelist:
                    trianglelist.append(triangle)
        else:
            sl = [s1tuples[l][1] for l in range(len(s1tuples))]
            pairs = itertools.combinations(sl, 2)
            rightpair = [pair for pair in pairs
                         if (min(pair[0],pair[1]),max(pair[0],pair[1])) in NNList[0]]
            if rightpair:
                rightpair = rightpair[0]
                triangle = [(s1, rightpair[0]),(rightpair[0],rightpair[1]),(s1, rightpair[1])]
                if triangle not in trianglelist:
                    trianglelist.append(triangle)
    return trianglelist


# In[ ]:


def LoadSpinConfigs(L,n,spinconfigfile):
    assert(n==1)

    s_ijl, ijl_s = kf.createspinsitetable(L[0])
    s42_ijl, ijl_s42, s42_realpos, s42_pos, pos_s42, configlist = OBCmapping(L[0], s_ijl, spinconfigfile)

    realspinstates = configlist[:,2:]
    return s42_ijl, ijl_s42, s42_realpos, s42_pos, pos_s42, realspinstates


# In[ ]:


def KagomeLatticeCharges(NNList, sconf, x, y):
    trianglelist = KagomeLatticeTriangles(NNList,len(sconf))
    chargeconf = []
    xc = []
    yc = []
    for triangle in trianglelist:
        s1 = triangle[0][0]
        s2 = triangle[1][0]
        s3 = triangle[2][1]

        chargeconf.append(sconf[s1] + sconf[s2] + sconf[s3])
        xc.append((x[s1] + x[s2] + x[s3])/3)
        yc.append((y[s1] + y[s2] + y[s3])/3)

    return np.array(chargeconf), np.array(xc), np.array(yc), trianglelist


# In[ ]:


def OBCmapping(L, s_ijl, filename):
    s42_ijl = []
    ijl_s42 = {}
    s42_realpos = []
    s42_pos = []
    pos_s42 ={}
    configlist = np.loadtxt(filename,delimiter=',')
    if L == 3:
        s42_ijl = [(0,3,1), (0,2,1),
                   (0,4,0), (1,3,2), (0,3,0), (1,2,2), (0,2,0), (1,1,2),
                   (1,3,1), (1,2,1), (1,1,1),
                   (1,4,0), (2,3,2), (1,3,0), (2,2,2), (1,2,0), (2,1,2), (1,1,0), (2,0,2),
                   (2,3,1), (2,2,1), (2,1,1), (2,0,1),
                   (3,3,2), (2,3,0), (3,2,2), (2,2,0), (3,1,2), (2,1,0), (3,0,2), (2,0,0),
                   (3,2,1), (3,1,1), (3,0,1),
                   (4,2,2), (3,2,0), (4,1,2), (3,1,0), (4,0,2), (3,0,0),
                   (4,1,1), (4,0,1)]
        for s42, (i,j,l) in enumerate(s42_ijl):
            ijl_s42[(i,j,l)] = s42
            s42_realpos.append(np.array((configlist[s42][0], configlist[s42][1])))
            x = (i + j/2.0)
            y = j * np.sqrt(3) /2.0
            if l == 0:
                x += 1 / 2.0
            if l == 1:
                x += 1 / 4.0
                y += np.sqrt(3) / 4.0
            if l == 2:
                x -= 1 / 4.0
                y += np.sqrt(3) / 4.0
            s42_pos.append(np.array((x,y)))
    return s42_ijl, ijl_s42, s42_realpos, s42_pos, pos_s42, configlist

