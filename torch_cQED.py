import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from scipy.stats import unitary_group

def raising_operator(dim):
    matrix = np.zeros([dim,dim]) *1j
    for i in range(dim-1):
        matrix[i+1][i] = np.sqrt(i+1)
    return matrix

def lowering_operator(dim):
    matrix = np.zeros([dim,dim])* 1j
    for i in range(dim-1):
        matrix[i][i+1] = np.sqrt(i+1)
    return matrix

def Hstatic(dim_c, dim_q, w_t, w_c, K, A, x, X):
    ar = raising_operator(dim_c)
    al = lowering_operator(dim_c)
    br = raising_operator(dim_q)
    bl = lowering_operator(dim_q)
    
    H_o = w_c*np.kron(ar@al,np.identity(dim_q)) + K/2*np.kron(ar@ar@al@al,np.identity(dim_q))
    H_t = w_t*np.kron(np.identity(dim_c),br@bl) + A/2*np.kron(np.identity(dim_c),br@br@bl@bl)
    H_i = x*np.kron(ar@al,br@bl)+X/2*np.kron(ar@ar@al@al,br@bl,)
    Hstatic = H_o + H_t + H_i
    return Hstatic

def cQED_hamiltonian_terms(dim_c, dim_q, w_t, w_c, K, A, x, X):
    ar = raising_operator(dim_c)
    al = lowering_operator(dim_c)
    br = raising_operator(dim_q)
    bl = lowering_operator(dim_q)
    
    h_1  = np.kron(al, np.identity(dim_q))+np.kron(ar, np.identity(dim_q))
    h_2 = np.kron(np.identity(dim_c),bl)+np.kron(np.identity(dim_c),br)
    h_1c  = ((np.kron(al, np.identity(dim_q)))-np.kron(ar, np.identity(dim_q)))*1j
    h_2c = ((np.kron(np.identity(dim_c),bl))-np.kron(np.identity(dim_c),br))*1j 
    h_d = Hstatic(dim_c, dim_q, w_t, w_c, K, A, x, X)
    return h_1, h_2, h_1c, h_2c, h_d


def hamiltonian(w1, w1_c, w2, w2_c):
    # Omega_c: cavity frequency
    # Omega_a: atom frequency
    # g: coupling strength
    H_1 = w1 * torch.tensor(h_1*10**6, dtype=torch.complex128,)
    H_1c = w1_c * torch.tensor(h_1c*10**6 , dtype=torch.complex128)
    H_2 = w2 * torch.tensor(h_2*10**6, dtype=torch.complex128)
    H_2c = w2_c * torch.tensor(h_2c*10**6, dtype=torch.complex128)

    H = H_1 + H_1c + H_2 + H_2c + torch.tensor(h_d)
    return H

def projector(dim_c, dim_b):
    p = torch.zeros([dim_c*2, dim_c*2],dtype=torch.complex128,)
    for i in range (2*dim_b, 2*dim_c):
        p[i][i] = dim_c/(dim_c-dim_b) # normalize to match the fid computer 
    return p

def evaluate_mps_mpo(unitary, mpo , L):
    
    # assume this is a translational invariant MPS
    isometry = unitary.reshape(dim_c,2,dim_c,2) # dim_c,dim_q
    
    # define boundary condition 
    bdry_mps = torch.eye(dim_c,dtype=torch.complex128)[0] 
    bdry_mpo_l = torch.eye(5,dtype = torch.complex128)[-1]#shape (2,5,2,5)
    bdry_mpo_r = torch.eye(5,dtype = torch.complex128)[0]
    
    # Contract input state with the first isometry
    mps_state = torch.einsum("a, abcd, i, ilkj, o, jrdo, bl-> ckr", bdry_mps, isometry, bdry_mps, torch.conj(isometry), bdry_mpo_l, mpo[0],torch.tensor([[1,0],[0,0]],dtype = torch.complex128))

    # Contract the MPS state with the MPO and the remaining isometries
    for i in range(1, L):
        mps_state = torch.einsum("aip, abcd, ilkj, jrdp, bl-> ckr", mps_state,  isometry, torch.conj(isometry), mpo[i],torch.tensor([[1,0],[0,0]],dtype = torch.complex128))

    # Contract the final MPS tensor with its conjugate
    result = torch.real(torch.einsum("aap, p-> ", mps_state,bdry_mpo_r))
    return result


from hamiltonian import model_mpo

L, burn_in = 30,20
J, V, h = 1, .5, -1

ham = torch.tensor(model_mpo.sd_ising(J, V, h, L - burn_in),dtype=torch.complex128)

def create_mpo(L, ham, burn_in = 0):
    mpo = []
    for _ in range (0, burn_in):
        mpo.append(torch.eye(10,dtype=torch.complex128).reshape(2,5,2,5))
    for _ in range (burn_in, L):
        mpo.append(ham)
    return mpo

def objective_function(parameters, dt, n_steps):
    w1, w1_c, w2, w2_c = parameters
    
    U = torch.eye(dim_c*dim_q, dtype=torch.complex128, requires_grad=True)  # Initialize the unitary
    for i in range(n_steps):
        H = hamiltonian(w1[i], w1_c[i], w2[i], w2_c[i])
        U = U@torch.matrix_exp(-1j * H * dt)
    #print(U)
    ##loss = 1 - fidelity
    
    penalty = 1 - torch.abs(torch.trace(torch.matmul(projector(dim_c, dim_b), U))) / (dim_c*dim_q)
    mpo = create_mpo(L, ham, burn_in)
    if dim_c != dim_b:
        loss = evaluate_mps_mpo(U, mpo , L) + penalty
    else:
        loss = evaluate_mps_mpo(U, mpo , L)
    
    return loss



dim_b = 4 # bond dimension = useful cavity levels
dim_c = dim_b+6  # of cavity/oscillator levels
dim_q = 2 # of qubit levels; which is usually set to 2

#System Parameter; taken from Ameya's 
w_c = 0#2*np.pi*4452*10**6
w_t = 0#2*np.pi*5664*10**6
K = -2*np.pi*3.7*1000
A = -2*np.pi*236*10**6
x = -2*np.pi*2.139 *10**6#chi
X = -2*np.pi*19*1000

h_1, h_2, h_1c, h_2c, h_d = cQED_hamiltonian_terms(dim_c, dim_q, w_t, w_c, K, A, x, X)

# Time step and number of steps
dt = 10 * 10**(-9)
n_steps = 200

# Initial parameters
w1 = Variable(torch.tensor(np.random.random(n_steps), dtype=torch.float64), requires_grad=True)
w1_c = Variable(torch.tensor(np.random.random(n_steps) , dtype=torch.float64), requires_grad=True)
w2 = Variable(torch.tensor(np.random.random(n_steps), dtype=torch.float64), requires_grad=True)
w2_c = Variable(torch.tensor(np.random.random(n_steps) , dtype=torch.float64), requires_grad=True)

# Target unitary
U_target = torch.tensor(unitary_group.rvs(dim_c*dim_q), dtype=torch.complex128)

# Set up the optimizer
optimizer = optim.Adam([w1, w1_c, w2, w2_c], lr=0.1)

n_epochs = 2000
import time
t0 = time.time()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = objective_function([w1, w1_c, w2, w2_c], dt, n_steps)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for param in [w1, w1_c, w2, w2_c]:
            param.clamp_(-10, 10)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss:{loss}, Time:{time.time()-t0}')
        
        
dim_b = 4 # bond dimension = useful cavity levels
dim_c = dim_b+16  # of cavity/oscillator levels
dim_q = 2 # of qubit levels; which is usually set to 2

h_1, h_2, h_1c, h_2c, h_d = cQED_hamiltonian_terms(dim_c, dim_q, w_t, w_c, K, A, x, X)
U = torch.eye(dim_c*dim_q, dtype=torch.complex128, requires_grad=True)  # Initialize the unitary
for i in range(n_steps):
    H = hamiltonian(w1[i], w1_c[i], w2[i], w2_c[i])
    U = U@torch.matrix_exp(-1j * H * dt)

mpo = create_mpo(L, ham, burn_in)
evaluate_mps_mpo(U, mpo , L)
