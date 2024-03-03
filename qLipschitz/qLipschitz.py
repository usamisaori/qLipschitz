#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# ## Support Functions

# In[12]:


from qiskit.quantum_info import DensityMatrix, Statevector

def getDensityMatrix(circuit):
    return DensityMatrix(circuit).data

def getStatevector(circuit):
    return Statevector(circuit).data


# In[11]:


from functools import reduce

Dag = lambda matrix: matrix.conj().T
Kron = lambda *matrices: reduce(np.kron, matrices)


# In[10]:


def powerSets(items):
    N = len(items)
    combs = []
    
    for i in range(2 ** N):
        comb = []
        for j in range(N):
            if (i >> j) % 2:
                comb.append(items[j])
        combs.append(comb)
    
    return combs 


# ## Measurements Operators

# In[4]:


# Measurements
psi_0 = np.array([1.0, 0.0])
psi_1 = np.array([0.0, 1.0])
I = np.eye(2)

M_0 = psi_0.reshape([2, 1]) @ psi_0.reshape([1, 2]).conj()
M_1 = psi_1.reshape([2, 1]) @ psi_1.reshape([1, 2]).conj()


# In[6]:


def getMeasurements(qubits_num):
    measurement_0 = [M_0]
    measurement_1 = [M_1]
    
    for i in range(qubits_num - 1):
        measurement_0.append(I)
        measurement_1.append(I)
        
    return [
        Kron(*measurement_0),
        Kron(*measurement_1)
    ]


# ## Algorithm Unit

# In[8]:


class Algorithm:
    def __init__(self, model_circuit, measurements, outputs):
        # DensityMatrix of model
        self.E = getDensityMatrix(model_circuit)
        
        # Measurements
        self.M = dict()
        for index, output in enumerate(outputs):
            self.M[output] = measurements[index]
        
        # Outputs
        self.O = outputs
        self.O_ = powerSets(outputs)


# ## Calculate Lipschitz Constant

# In[9]:


def qLipschitz(A):
    E, M, O, O_ = A.E, A.M, A.O, A.O_
    
    # Step 1: Calculate W_i
    W = dict()
    for i in O:
        W[i] = Dag(E) @ Dag(M[i]) @ M[i] @ E
    
    # Step 2: Calculate K_star
    K_star = 0; vectors = [None, None]
    M_star = np.zeros(E.shape)
    
    for S in O_:
        if len(S) == 0:
            continue
            
        # calculate M_S = Σ Wi
        M_S = np.zeros(E.shape).astype('complex64')
        for i in S:
            M_S += W[i]
        
        # calculate eigenvalues and eigenvectors of M_S
        eigenvalues, eigenvectors = np.linalg.eigh(M_S)
        min_index = np.where(eigenvalues == eigenvalues.min())
        max_index = np.where(eigenvalues == eigenvalues.max())
        
        # calculate K_S
        K_S = np.linalg.norm(eigenvalues[max_index][0] - eigenvalues[min_index][0])
        
        if K_S > K_star:
            K_star = K_S
            vectors[0] = eigenvectors.T[max_index][0]
            vectors[1] = eigenvectors.T[min_index][0]
            
    return K_star, np.array(vectors)


# ## Fairness verifying

# In[13]:


def FairVeriQ(A, epsilon, delta):
    # epsilon <= 1 and delta > 0
    K_star, kernel = Lipschitz(A)
    
    if delta >= K_star * epsilon:
        return True, None
    else:
        return False, kernel


# ## Generate Bias pair

# In[14]:


def generateBiasPair(sigma, kernel, epsilon):
    psi, phi = kernel
    size = len(psi)
    psi = psi.reshape(size, 1) @ Dag(psi.reshape(size, 1))
    phi = phi.reshape(size, 1) @ Dag(phi.reshape(size, 1))
    
    rou_psi = epsilon * psi + (1 - epsilon) * sigma
    rou_phi = epsilon * phi + (1 - epsilon) * sigma
    
    return np.array([
        rou_psi, rou_phi
    ])


# ## Encapsulated qLipschitz Class

# In[17]:


class QLipschitz:
    def __init__(self, model_circuit, outputs):
        measurements = getMeasurements(model_circuit.num_qubits)
        self.A = Algorithm(model_circuit, measurements, outputs)
        
    @property
    def constant(self):
        return qLipschitz(self.A)
    
    def fairVeriQ(self, epsilon, delta):
        return FairVeriQ(self.A, epsilon, delta)
    
    def generateBiasPair(self, sigma, kernel, epsilon):
        return generateBiasPair(sigma, kernel, epsilon)


# In[18]:

