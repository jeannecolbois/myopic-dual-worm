
# coding: utf-8

# In[ ]:


from functools import lru_cache
import GraphDrawing as gdw


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


# In[ ]:


def reducedgraphdice(L, a, d_ijl, v_ijl, ijl_v):
    #table of where to look at 
    nv = [(0, -1, 2), (0, 0, 1), (0, 0, 2), (-1, 1, 1), (-1, 0, 2), (-1, 0, 1)]
    #dimer -> vertex: l from 0 to 5 indicates the edge
    d_2v = [((i, j, 0),(i + nv[l][0], j + nv[l][1], nv[l][2])) for (i, j, l) in d_ijl]
    d_2v = [(ijl_v[i, j, l], ijl_v[fixbc(ni, nj, nl, L)]) for ((i, j, l), (ni, nj, nl)) in d_2v]
    
    v2_d = {} 
    for d, (v1,v2) in enumerate(d_2v):
        v2_d[v1,v2] = d
        v2_d[v2,v1] = d
    
    return d_2v, v2_d


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def plot_dice_nodes(L, a, color='black', s=1, **kargs):
    (v_ijl, ijl_v, e_2v, pos) = graphdice(L, a)
    gdw.draw_nodes(pos, list(pos.keys()), c=color, s=s, **kargs)
def plot_dice_dimers(dimers, d_ijl, L, a, **kargs):
    """
        :param dimers: list of dimers indices (for instance [3, 5, 2, ...])
    """
    (v_ijl, ijl_v, e_2v, pos) = graphdice(L, a)
    e_d = edge2dimer(L, a, d_ijl, v_ijl, ijl_v, e_2v)
    dimersthere = [] # list of vertex pairs on the dice lattice where there is a dimer
    for e, (v1, v2) in enumerate(e_2v):
        #get corresponding state
        if e_d[e] in dimers:
            dimersthere.append((v1,v2))
    gdw.draw_edges(pos, dimersthere, **kargs)
def plot_dice_dimerstate(state, d_ijl, L, a, node_color = 'black', dim_color = 'black', no_dim_color = 'lightgrey', linewidth = 1, **kargs):
    plot_dice_nodes(L, a, color = node_color, **kargs)
    dimers = [d for d, dstate in enumerate(state) if dstate == 1]
    nodimers = [d for d, dstate in enumerate(state) if dstate == -1]
    plot_dice_dimers(dimers, d_ijl, L, a, color = dim_color, linewidth = linewidth)
    plot_dice_dimers(nodimers, d_ijl, L, a, color = no_dim_color)


# In[ ]:


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


# In[ ]:


def plot_kag_nodes(L, a, color='blue', s=20, **kargs):
    (sv_ijl, ijl_sv, e_2sv, poskag) = graphkag(L,a)
    gdw.draw_nodes(poskag, list(poskag.keys()), c = color, s = s, **kargs)

def plot_kag_edges(L, a, color='lightblue', **kargs):
    sv_ijl, ijl_sv, e_2sv, pos = graphkag(L, a)
    gdw.draw_edges(pos, e_2sv, **kargs)
    
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
    gdw.draw_nodes(poskag, spinsthere, c = color, **kargs)
    
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


# In[ ]:


#FUNCTION ALLOWING TO PLOT THE FULL STATE

def plotstate(temp_id, L, d_ijl, ijl_s, sidlist, didlist, s2_d, states, dim_node_color = 'black', dim_color ='black', no_dim_color = 'lightgrey', spin_edge_color = 'lightblue', spin_up_color = 'blue', spin_down_color = 'red', dimerlinewidth = 5, spinlinewidth = 1, **kargs):    

    a = 2 #lattice parameter
    spinstate = onestate_dimers2spins(sidlist, didlist, L, states[temp_id])
    plt.figure(figsize = (2*L,2*L))
    plt.axis('equal')
    plot_dice_dimerstate(states[temp_id], d_ijl, L, a, dim_node_color, dim_color, no_dim_color, linewidth = dimerlinewidth, **kargs)
    plot_kag_spinstate(spinstate, ijl_s, L, a, spin_edge_color, spin_up_color, spin_down_color, linewidth = spinlinewidth, **kargs)
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = 'off',      # ticks along the bottom edge are off
        top = 'off',         # ticks along the top edge are off
        labelbottom = 'off',
        left = 'off',
        right = 'off',
        labelleft = 'off') # labels along the bottom edge are off


# In[ ]:


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
    
    gdw.draw_function(pos, new_f, list(pos.keys()), s = s, marker = 'h', **kargs)
    
    plt.axis('equal')
    #plt.axis([0, 1, 0, 3*n])
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = 'off',      # ticks along the bottom edge are off
        top = 'off',         # ticks along the top edge are off
        labelbottom = 'off',
        left = 'off',
        right = 'off',
        labelleft = 'off') # labels along the bottom edge are off


# In[ ]:


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


# In[ ]:


def plot_reciprocal(L, a, n, color = 'blue', **kargs):
    #get the shape of the lattice
    (qv_k1k2, k1k2_qv, pos, factor) = reciprocalgraph(L, a)
    
    #chose the size of the points in a way that compensates for the parameters
    s = (48/n*factor) ** 2
    
    #draw the lattice
    gdw.draw_nodes(pos, list(pos.keys()), c = color, edgecolors = 'none', s = s, marker = 'H', **kargs)
    
    #make it a nice plot
    plt.axis('equal')
    plt.axis([0, 1, -3*n, 3*n])
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = 'off',      # ticks along the bottom edge are off
        top = 'off',         # ticks along the top edge are off
        labelbottom = 'off',
        left = 'off',
        right = 'off',
        labelleft = 'off') # labels along the bottom edge are off  


# In[ ]:


def plot_function_reciprocal(f, L, a, **kargs):
    '''
        Plots the function f over the reciprocal lattice. f has to be defined on the "square" brillouin zone
        for k1, k2 in range(0, L), which will be the usual output of the FT
    '''
    #get the shape of the lattice
    (qv_k1k2, k1k2_qv, pos, factor) = reciprocalgraph(L, a) # drawing lattice
    
    s =(110/L)**2
    gdw.draw_function(pos, f, list(pos.keys()), s = s, **kargs)
    
    plt.axis('equal')
   # plt.axis([0, 1, -10, 10])
    plt.tick_params(  
        which = 'both',      # both major and minor ticks are affected
        bottom = 'off',      # ticks along the bottom edge are off
        top = 'off',         # ticks along the top edge are off
        labelbottom = 'off',
        left = 'off',
        right = 'off',
        labelleft = 'off') # labels along the bottom edge are offb

