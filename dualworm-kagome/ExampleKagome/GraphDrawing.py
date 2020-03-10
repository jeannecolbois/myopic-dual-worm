
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt


# In[ ]:


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


# In[ ]:


def draw_edges(pos, edgelist, *args, **kargs):
    '''
        This function draws the edges of a graph
    '''
    ax = plt.gca()

    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    edge_collection = LineCollection(edge_pos, *args, **kargs)
    edge_collection.set_zorder(1)  # edges go behind nodes
    ax.add_collection(edge_collection)


# In[ ]:


def draw_function(pos, function, nodelist, marker = 'H', *args, **kargs):
    '''
        This function draws a function over a graph
    '''
    #create the colormap according to the value of the function
    color = [function[v] for v in nodelist]
    #draw the nodes using the colormap
    draw_nodes(pos, nodelist, c = color, edgecolors = 'none', cmap = 'plasma', marker = marker, **kargs)

