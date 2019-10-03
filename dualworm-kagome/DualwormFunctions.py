
# coding: utf-8

# In[ ]:

import numpy as np
import dimers as dim
import CleanKagomeFunctions as lattice


# In[ ]:

def createdualtable(L):
    '''
        Creates the table of dual bonds corresponding to the dual lattice of side size L.
        Returns a table identifing an int with the three coordinates of the dual bond and a dictionnary identifying the
        three coordinates with the dual bond's int index. This allows to handle other relations between dual bonds in an
        easier way.
    '''
    return lattice.createdualtable(L)


# In[ ]:



