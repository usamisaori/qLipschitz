#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.linalg import sqrtm


# In[4]:


Dag = lambda matrix: matrix.conj().T


# In[5]:


def D(rou, sigma):
    A = rou - sigma
    A_ = sqrtm( np.dot( A.conj().T, A ) )

    return 0.5 * np.linalg.norm( np.trace(A_) )


# In[6]:


def d(A, rou, sigma):
    distance = 0
    
    for output in A.O:
        trace = np.trace(
            Dag(A.M[output]) @ A.M[output] @ 
            (A.E @ (rou - sigma) @ Dag(A.E))
        )
        
        distance += np.linalg.norm(trace)
    
    return distance / 2


# In[ ]:




