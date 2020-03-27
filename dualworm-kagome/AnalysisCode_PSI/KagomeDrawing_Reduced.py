
# coding: utf-8

# Last update 18.02.2019
# Author : Jeanne Colbois
# Please send any comments, questions or remarks to Jeanne Colbois: jeanne.colbois@epfl.ch.
# The author would appreciate to be cited in uses of this code, and would be very happy to hear about potential nice developments.

# In[1]:


import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt


# In[2]:


#############################################################
                ## BASES FOR DRAWING ##
#############################################################


# In[3]:


# create a function to fix the periodic boundary conditions
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


# In[4]:


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
        dim_id = didlist[it]
        spin_id = sidlist[it+1]
        s = spinstate[spin_id] = s * state[dim_id]
    return spinstate


# In[5]:


#Function creating a list of vertices corresponding to the kagome
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


# In[8]:


#############################################################
                ## DRAW A GRAPH ##
#############################################################


# In[9]:


### FUNCTIONS DRAWING EITHER NODES OR EDGES OF A GRAPH
def draw_nodes(pos, nodelist, c, **kargs):
    '''
        This function draws the nodes of a graph
    '''
    
    ax = plt.gca()

    xy = np.asarray([pos[v] for v in nodelist])
    node_collection = plt.scatter(xy[:, 0], xy[:, 1], c = c, **kargs)
    if isinstance(c, list):
        plt.colorbar()
    node_collection.set_zorder(2)

def draw_edges(pos, edgelist, c = 'blue', vmin = -1, vmax = 1,*args, **kargs):
    '''
        This function draws the edges of a graph
    '''
    ax = plt.gca() # get current axis
    fig = plt.gcf() # get current figure
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    if isinstance(c,list):
        c += [vmin, vmax]## adding two values at the end of the table doesn't change what we plot
        colors = np.array(c) 
        edge_collection = LineCollection(edge_pos, array = colors, cmap = plt.cm.plasma, *args, **kargs)
        edge_collection.set_zorder(1)  # edges go behind nodes
        ax.add_collection(edge_collection)
        fig.colorbar(edge_collection)
    else:
        edge_collection = LineCollection(edge_pos, *args, **kargs)
        edge_collection.set_zorder(1)  # edges go behind nodes
        ax.add_collection(edge_collection)

# In[10]:


#############################################################
                ## KAGOME LATTICE STATE ##
#############################################################


# In[14]:


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


# In[15]:


def plot_kag_nodes(L, a, color='blue', s=20, **kargs):
    (sv_ijl, ijl_sv, e_2sv, poskag) = graphkag(L,a)
    draw_nodes(poskag, list(poskag.keys()), c = color, s = s, **kargs)

def plot_kag_edges(L, a, color='lightblue', **kargs):
    sv_ijl, ijl_sv, e_2sv, pos = graphkag(L, a)
    draw_edges(pos, e_2sv, **kargs)
    
def plot_kag_spins(spins, ijl_s, L, a, color = 'red', **kargs):
    """
        :param spins: list of spins indices (for instance [3, 5, 2, ...])
    """
    (sv_ijl, ijl_sv, e_2sv, poskag) = graphkag(L,a)
    sv_s = spinvertex2spin(L,a, ijl_s, sv_ijl)
    spinsthere = []
    for sv, ijl in enumerate(sv_ijl): #for each key
        if sv_s[sv] in spins:# look for corresponding spin in spins
            spinsthere.append(sv) #list of keys
    draw_nodes(poskag, spinsthere, c = color, **kargs)
    
def plot_kag_spinstate(spinstate, ijl_s, L, a, edge_color = 'lightblue', up_color = 'blue', down_color = 'red', **kargs):
    """
        :param spins: list of spins indices (for instance [3, 5, 2, ...])
    """
    plot_kag_nodes(L, a, **kargs)
    plot_kag_edges(L, a, color = edge_color, **kargs)
    spinsup = [s for s, sstate in enumerate(spinstate) if sstate == 1]
    spinsdown = [s for s, sstate in enumerate(spinstate) if sstate == -1]
    if len(spinsup) != 0:
        plot_kag_spins(spinsup, ijl_s, L, a, color = up_color, label = 'spin up')
    if len(spinsdown) != 0:
        plot_kag_spins(spinsdown, ijl_s, L, a, color = down_color, label = 'spin down')


# In[16]:

### SPIN SITES TABLE
# make the mapping between the spinsite index s\in {0,..., S-1} and the numbering (ijl)
def createspinsitetable(L):
    '''
        Creates the table of spin sites corresponding to a dice lattice of side size L.
        Returns a table identifing an int with the three coordinates of the spin site and a dictionnary identifying the
        three coordinates with the spin site's int index. This allows to handle other relations between spin sites in an
        easier way.
    '''
    s_ijl = [(i, j, l) for i in range(2*L) for j in range(2*L) for l in range(3) if (i+j > L-2) and (i+j < 3*L-1)]
    # dictionary
    ijl_s = {}
    for s, triplet in enumerate(s_ijl):
        ijl_s[triplet] = s
    return s_ijl, ijl_s


# In[18]:


def plot_function_kag(f, L, a, **kargs):
    '''
        Plots the function f over the kagomé lattice. f has to be defined on the "square" brillouin zone
        for k1, k2 in range(0, L), which will be the usual output of the FT
    '''
    #get the shape of the lattice
    (sv_ijl, ijl_sv, e_2sv, pos) = graphkag(L, a) # drawing lattice
    (s_ijl, ijl_s) = createspinsitetable(L) # lattice over which the function is defined
    
    
    #chose the size of the points in a way that compensates for the parameters
    s = (a*54/L)**2
    
    new_f = np.zeros(len(sv_ijl))
    for sv, (i, j, l) in enumerate(sv_ijl):
        new_f[sv] = f[ijl_s[fixbc(i, j, l, L)]]
    
    draw_function(pos, new_f, list(pos.keys()), s = s, marker = 'h', **kargs)
    
    plt.axis('equal')
    #plt.axis([0, 1, 0, 3*n])
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = False,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = False,
        left = False,
        right = False,
        labelleft = False) # labels along the bottom edge are off


# In[19]:


#############################################################
        ## PLOTS ON THE RECIPROCAL LATTICE ##
#############################################################


# In[20]:


def KagomeSimpleReciprocal(L):
    '''
        Creates the table of reciprocal space points where there can be a non-zero amplitude of the diffraciton signal.
        Returns a table identifying an int with the two reciprocal space indices and a dictionary identifying the two
        indices with the corresponding int.
    '''
    q_k1k2 = [(k1, k2) for k1 in range(L) for k2 in range(L)]
    k1k2_q = {} # new empty dictionary
    for q, couple in enumerate(q_k1k2):
        k1k2_q[couple] = q
    return q_k1k2, k1k2_q


# In[ ]:


def KagomeReciprocal(L):
    '''
        Creates the table of reciprocal space points where there can be a non-zero amplitude of the diffraciton signal.
        Returns a table identifying an int with the two reciprocal space indices and a dictionary identifying the two
        indices with the corresponding int.
    '''
    #possible values of q: 
    q_k1k2 = [(k1, k2) for k1 in range(-2*L, 2*L) for k2 in range(-4*L, 4*L) if (abs(2 * k1 - k2) <= 2*L and abs(2*k2 - k1) <= 2*L and abs(k1 + k2) <= 2*L)]
    k1k2_q = {} # new empty dictionary
    for q, couple in enumerate(q_k1k2):
        k1k2_q[couple] = q
    return q_k1k2, k1k2_q


# In[21]:


#Function creating a list of vertices corresponding to the triangular lattice
from functools import lru_cache

@lru_cache(maxsize = None)
def reciprocalgraph(L, a):
    '''
        For the triangular lattice:
        Returns two vertex <-> (k1, k2) tables,  and a dictionary giving the position (x,y) of each vertex
    '''
    (qv_k1k2, k1k2_qv) = KagomeReciprocal(L)
    
   
    #position
    pos = {} #empty dictionary
    factor = 2 * np.pi / (a*L)
    for qv, (k1, k2) in enumerate(qv_k1k2):
        x = factor * k1
        y = factor / np.sqrt(3) * (2 * k2 - k1)
        pos[qv] = (x,y)
        
    return qv_k1k2, k1k2_qv, pos, factor


# In[22]:


def draw_function(pos, function, nodelist, marker = 'H', *args, **kargs):
    '''
        This function draws a function over a graph
    '''
    #create the colormap according to the value of the function
    color = [function[v] for v in nodelist]
    #draw the nodes using the colormap
    draw_nodes(pos, nodelist, c = color, edgecolors = 'none', cmap = 'plasma', marker = marker, **kargs)


# In[23]:


def plot_reciprocal(L, a, n, color = 'blue', **kargs):
    #get the shape of the lattice
    (qv_k1k2, k1k2_qv, pos, factor) = reciprocalgraph(L, a)
    
    #chose the size of the points in a way that compensates for the parameters
    s = (48/n*factor) ** 2
    
    #draw the lattice
    draw_nodes(pos, list(pos.keys()), c = color, edgecolors = 'none', s = s, marker = 'H', **kargs)
    
    #make it a nice plot
    plt.axis('equal')
    plt.axis([0, 1, -3*n, 3*n])
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = False,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = False,
        left = False,
        right = False,
        labelleft = False) # labels along the bottom edge are off  


# In[24]:


def plot_function_reciprocal(f, L, a, **kargs):
    '''
        Plots the function f over the reciprocal lattice. f has to be defined on the "square" brillouin zone
        for k1, k2 in range(0, L), which will be the usual output of the FT
    '''
    #get the shape of the lattice
    (qv_k1k2, k1k2_qv, pos, factor) = reciprocalgraph(L, a) # drawing lattice
    
    s =(110/L)**2
    draw_function(pos, f, list(pos.keys()), s = s, **kargs)
    
    plt.axis('equal')
    # plt.axis([0, 1, -10, 10])
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = False,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = False,
        left = False,
        right = False,
        labelleft = False) # labels along the bottom edge are off

############################################################
#                     CORRELATIONS                         #
############################################################
def draw_values_kagedges(pos, listedges, values, vmin = -1, vmax = 1, nodescolor = 'blue', nodeslinewidth = 2, edgelinewidth = 2, **kargs):
    '''
        Plots the values over the edges in list edges, according to the list of corresponding positions pos.
    '''
    plt.figure(dpi=200)
    draw_nodes(pos, [s for s in range(len(pos))], c = nodescolor)
    draw_edges(pos, listedges, c = values, vmin = vmin, vmax = vmax, linewidth = edgelinewidth)
    plt.axis('equal')
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = False,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = False,
        left = False,
        right = False,
        labelleft = False) # labels along the bottom edge are off

    
    