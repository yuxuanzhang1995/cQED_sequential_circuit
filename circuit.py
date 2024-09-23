import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
import numpy as np
from functools import partial

"""
Code interfaces with this module entirely through the Circuit class. A Circuit
represents a gate circuit, and contains information about the qudit register,
the gates, and the parameters that parametrize the gates. Evaluation of the
circuit (given a parameter vector) returns the corresponding unitary matrix.
Circuit evaluation is accelerated by a just-in-time compiler provided by Google
JAX (https://github.com/google/jax), so in theory it should be quick to
repeatedly evaluate the circuit during the VQE optimization.
"""

class Circuit:
    """
    Represents a quantum circuit.
    
    Attributes
    ----------
    regInfo: RegisterInfo
        Contains information about the dimension, shape, etc of the qudit register.
    gates: list
        The quantum gates comprising the circuit. After assembly, this consists of GateLayers
        which themselves contain the actual gates.
    n_params: int
        The number of free gate parameters in the circuit. Only set after the circuit is assembled.
    """
    
    def __init__(self, register):
        """
        Parameters
        ----------
        register: list
            A list of 3-tuples (str, str, int) with each entry describing a quantum resource.
            The first string must be one of "qubit" or "cavity", the second either "p" or "b"
            (corresponding to whether the resource is part of physical or bond legs), and
            the int is the number of levels in that qudit (i.e. 2 for a qubit). len(register)
            should equal the number of quantum resources.
        """
        self.regInfo = RegisterInfo(register)
        self.gates = []
        self.n_params = None
        
    def add_gate(self, gate_type, qids=[], n_params=None, fn=None):
        """
        Add a gate to the circuit. Note that the ordering of qids is important for
        multi-qudit gates. For a SNAP gate, the first qudit id refers to the qubit,
        and the second refers to the cavity.
        
        Parameters
        ----------
        gate_type: str
            One of "rotation", "displacement", or "snap".
        qids: list (optional)
            The qudit labels that this gate acts on. Quantum resources are labelled by their
            index in the register (starting at 0). If qids=[1,3,4], then this gate acts
            on qudits 1, 3, and 4. If qids can be uniquely inferred from the register for the
            given gate type, it is optional to specify. The order of qids for multi-qudit
            gates is important; i.e. a SNAP gate should specify qids = [qubit_id, cavity_id].
        n_params: int (optional)
            The number of free real-valued parameters this gate requires. An arbitrary
            qubit rotation requires 3 real params, whereas a fixed rotation around a fixed
            axis (e.g. an X gate) requires 0 free params. If this argument is unspecified,
            it takes the native value for the gate_type specified. The native gate parameters
            are documented for each gate. If this argument is specified, the fn argument must
            also be specified.
        fn: function (optional)
            A function whose input is a list slice (length=n_params) of the relevant
            parameters from the parameter vector, processes them, and outputs an ndarray that
            represents the native parameter values for the gate. If this argument is 
            unspecified, the native parameters used are taken directly from the parameter
            vector. The native gate parameters are documented for each gate. If this argument
            is specified, the n_params argument must also be specified.
        """
        gate_type = gate_type.lower().strip()
        reg_levels = self.regInfo.qudit_levels
        reg_types = self.regInfo.qudit_types
        if gate_type not in ["rotation", "displacement", "snap", "cp", "h", "cpm"]:
            assert False, "Invalid gate type."
        for qid in qids:
            assert type(qid)==int, "qids must be int"
            assert qid >= 0, "Invalid qid: {}".format(qid)
            assert qid < len(reg_levels), "Invalid qid: {}".format(qid)
        out = None
        if n_params!=None or fn!=None:
            assert n_params!=None and fn!=None, "Must specify both n_params and fn (or neither)"
            try:
                out = fn(np.zeros(n_params))
            except:
                assert False, "n_params doesn't match supplied fn"
        gate = None
        if gate_type == "rotation":
            if qids == []:
                assert reg_types.count("qubit") == 1, "Must specify rotation qubit qid"
                qids = [reg_types.index("qubit")]
            assert len(qids) == 1, "Rotation gate must act on exactly 1 qubit"
            assert reg_types[qids[0]] == "qubit", "Rotation qid {} is not a qubit".format(qid)
            if fn==None:
                gate = RotGate(qids)
            else:
                assert len(out) == 3, "fn for Rotation gate must output 3 parameters"
                gate = RotGate(qids, n_params, fn)
        elif gate_type == "cp":
            gate = CPGate(qids, n_params, fn)
        elif gate_type == "cpm":
            assert len(qids) == 2, "SNAP gate must act on exactly 2 qudits"
            assert reg_types[qids[0]] == "qubit", "SNAP qid {} is not a qubit".format(qid)
            assert reg_types[qids[1]] == "cavity", "SNAP qid {} is not a cavity".format(qid)
            dim = reg_levels[qids[0]] * reg_levels[qids[1]]
            gate = CPMGate(dim, qids, n_params, fn)
        elif gate_type == "h":
            if fn==None:
                gate = HGate(qids)
            else:
                gate = HGate(qids, n_params, fn)
        elif gate_type == "displacement":
            if qids == []:
                assert reg_types.count("cavity") == 1, "Must specify displacement cavity qid"
                qids = [reg_types.index("cavity")]
            assert len(qids) == 1, "Displacement gate must act on exactly 1 cavity"
            assert reg_types[qids[0]] == "cavity", "Displacement qid {} is not a cavity".format(qid)
            dim = reg_levels[qids[0]]
            if fn==None:
                gate = DispGate(dim, qids)
            else:
                assert len(out) == 2, "fn for Displacement gate must output 2 real parameters"
                gate = DispGate(dim, qids, n_params, fn)
        elif gate_type == "snap":
            if qids == []:
                assert reg_types.count("qubit") == 1, "Must specify SNAP qubit qid"
                assert reg_types.count("cavity") == 1, "Must specify SNAP cavity qid"
                qids = [reg_types.index("qubit"), reg_types.index("cavity")]
            assert len(qids) == 2, "SNAP gate must act on exactly 2 qudits"
            assert reg_types[qids[0]] == "qubit", "SNAP qid {} is not a qubit".format(qid)
            assert reg_types[qids[1]] == "cavity", "SNAP qid {} is not a cavity".format(qid)
            dim = reg_levels[qids[0]] * reg_levels[qids[1]]
            if fn==None:
                gate = SNAPGate(dim, qids)
            else:
                assert len(out) == dim//2, "fn for SNAP gate must output dim/2 parameters"
                gate = SNAPGate(dim, qids, n_params, fn)
        assert gate != None
        
        self.gates.append(gate)

    def assemble(self):
        """
        Assemble the circuit to make it ready for evaluation. Organizes the gates into
        layers using a simple non-overlap scheme, assigns each gate a subset of the
        parameter vector, and fills "incomplete" layers with identity gates.
        """
        # fill out the current layer and append it to the list of layers.
        def complete(layer):
            # use symmetric difference of sets to find unused qudits in this layer
            unused = used_ids ^ set(self.regInfo.ids)
            # fill the layer with identity gates for each unused qudit
            for qid in unused:
                dim = self.regInfo.qudit_levels[qid]
                layer.append(Gate(dim, [qid]))
            layers.append(GateLayer(gates=layer, regInfo=self.regInfo))
        
        layers = []
        # a counter that represents the next gate's index in the parameter vector
        param_ind = 0
        # all the gates that are going into the current layer
        layer = []
        # all the qudit ids that are being used in the current layer
        used_ids = set()
        for g in self.gates:
            g.param_ind = param_ind
            param_ind += g.n_params
            g_ids = set(g.qudit_ids)
            # check if the gate overlaps with any of the gates in the current layer
            # if so, fill out this layer, append it, and start a new layer
            if len(g_ids.intersection(used_ids)) != 0:
                complete(layer)
                used_ids = set()
                layer = []
            layer.append(g)
            used_ids = used_ids.union(g_ids)
        # take care of the last layer
        complete(layer)
        
        # gates are now the series of GateLayers
        self.gates = layers
        self.n_params = param_ind

    @partial(jit, static_argnums=(0,))
    def evaluate(self, params):
        """
        Evaluates the circuit and returns its unitary operator as a matrix. You must
        assemble the circuit before evaluating it. Uses a JIT compiler; for optimal
        performance, do not modify the circuit nor its constituent gates between calls
        to this method.
        
        Parameters
        ----------
        params: 1D ndarray
            A vector of parameters for the gates.
        
        Returns
        -------
        2D ndarray. A unitary matrix representing the circuit.
        """
        assert len(params) == self.n_params
        mat = jnp.eye(self.regInfo.dim)
        for layer in self.gates:
            g = layer.gate(params)
            # reverse since operators act right-to-left
            mat = jnp.matmul(g, mat)
        return mat
    
    def get_tensor(self, params):
        """
        Evaluates the circuit and returns its unitary operator as a 4-leg tensor of 
        shape (physical_dim, bond_dim, physical_dim, bond_dim). You must assemble the
        circuit before evaluating it. Uses a JIT compiler; for optimal performance, do
        not modify the circuit nor its constituent gates between calls to this method.
        
        Parameters
        ----------
        params: 1D ndarray
            A vector of parameters for the gates.
        
        Returns
        -------
        4D ndarray. Shape (physical_dim, bond_dim, physical_dim, bond_dim). A unitary
            matrix representing the circuit.
        """
        mat = self.evaluate(params)
        mat = mat.reshape(self.regInfo.shape)
        mat = jnp.moveaxis(mat, self.regInfo.tensor_permutes, self.regInfo.unpermuted)
        mat = mat.reshape(self.regInfo.tensor_shape)
        return mat
        

class RegisterInfo:
    """
    Information about a qudit register.
    
    Attributes
    ----------
    qudit_levels: list
        Each entry is an int corresponding to the local dimension (# of levels)
        in the corresponding quantum resource (i.e. 2 for a qubit).
    qudit_types: list
        Each entry is a str (one of either "qubit" or "cavity") encoding the type of
        each quantum resource
    dim: int
        The total Hilbert dimension of the register
    ids: list
        The labels for each of the quantum resources. Given by list(range(len(register)))
    shape: list
        The qudit-indexed shape of a unitary tensor acting on the register
    unpermuted: list
        The unpermuted indices of the unitary, in terms of qudits. Given by
        list(range(2 * len(register)))
    tensor_permutes: list
        The permutation of indices (of the unitary) required to organize the qudits into
        physical qudits followed by bond qudits. This is to help reshape into a 4-leg tensor
    tensor_shape: tuple
        The shape of the 4-leg tensor: (physical_dim, bond_dim, physical_dim, bond_dim)
    """
    
    def __init__(self, register_info):
        self.qudit_types = [x[0] for x in register_info]
        self.qudit_levels = [x[2] for x in register_info]
        for i in range(len(register_info)):
            if self.qudit_types[i]=='qubit':
                assert(self.qudit_levels[i]==2), "Qubit must have exactly 2 levels"
            elif self.qudit_types[i]=='cavity':
                assert(self.qudit_levels[i]>=2), "Cavity must have >=2 levels"
            else:
                assert False, "{} is not a valid qudit type".format(self.qudit_types[i])
        self.dim = np.prod(self.qudit_levels)
        self.ids = list(range(len(register_info)))
        
        self.shape = self.qudit_levels + self.qudit_levels
        self.unpermuted = list(range(2 * len(register_info)))
        phys_inds = [i for i in self.ids if register_info[i][1]=="p"]
        bond_inds = [i for i in self.ids if register_info[i][1]=="b"]
        permutes = phys_inds + bond_inds
        assert len(permutes) == len(register_info), "Every qudit must be designated either 'p' or 'b'"
        self.tensor_permutes = permutes + [len(permutes) + qid for qid in permutes]
        phys_dim = int(np.prod([self.qudit_levels[i] for i in phys_inds]))
        bond_dim = int(np.prod([self.qudit_levels[i] for i in bond_inds]))
        self.tensor_shape = (phys_dim, bond_dim, phys_dim, bond_dim)


class Gate:
    """
    Base class for all gates. By default, implements the identity matrix.
    The gate method returns the unitary matrix corresponding to this gate. All
    subclasses should override this method, and implementations should be as efficient
    as possible.
    """
    
    def __init__(self, dim, qids, n_params=0, fn=lambda x:x):
        """
        Parameters
        ----------
        dim: int
            The dimension of this gate. Should be the product of local dimensions for all
            qudits that this gate acts on.
        qids: list
            A list of the qudit labels (in the parent circuit) that this gate acts on
        n_params: int
            The number of free parameters this gate requires
        fn: function
            A function whose input is a list slice of the relevant parameters
            from the parameter vector, processes them, and outputs an ndarray that
            represents the actual parameters for the gate. Note that the length of the
            input list slice should match n_params.
        """
        self.dim = dim
        self.qudit_ids = qids
        self.n_params = n_params
        # param_ind: where in the parameter vector to look for relevant params
        # this is currently set during circuit assembly
        self.param_ind = None
        self.process = fn
        
    def extract(self, params):
        return params[self.param_ind:self.param_ind+self.n_params]
        
    def gate(self, params):
        """
        Parameters
        ----------
        params: 1D ndarray
            The full parameter vector. Each gate should use its param_ind to locate the
            relevant parameters.
        """
        return jnp.eye(self.dim)

class RotGate(Gate):
    """
    An arbitrary qubit rotation. Natively, it is parametrized by three real parameters:
    theta (the polar angle of the rotation axis), phi (azimuthal angle of rotation axis),
    and rotangle (the angle of rotation around the rotation axis). If fn is specified, it
    should return an unpackable object: theta, phi, rotangle = fn(param_vec_slice)
    """
    
    def __init__(self, qids, n_params=3, fn=lambda x:x):
        super().__init__(dim=2, qids=qids, n_params=n_params, fn=fn)
        self.paulix = jnp.array([[0, 1],[1, 0]])
        self.pauliy = jnp.array([[0,-1j],[1j, 0]])
        self.pauliz = jnp.array([[1, 0],[0,-1]])            

    def gate(self, params):
        # get relevant params from param vector
        theta, phi, rotangle = self.process(self.extract(params))
        # use cayley-hamilton theorem for variant form of rotation operator
        # https://arxiv.org/pdf/1402.3541.pdf
        # sin term has minus sign bc operator is exp(-i theta nS)
        inS = 1j * (jnp.sin(theta)*jnp.cos(phi)*self.paulix + \
                    jnp.sin(theta)*jnp.sin(phi)*self.pauliy + \
                    jnp.cos(theta)*self.pauliz)
        return jnp.cos(rotangle/2)*jnp.eye(2) - jnp.sin(rotangle/2)*inS
    
class HGate(Gate):
    def __init__(self, qids, n_params=0, fn=lambda x:x):
        super().__init__(dim=2, qids=qids, n_params=n_params, fn=fn)
        self.H1 = jnp.array([[1,1],[1,-1]])

    def gate(self, params):
        # get relevant params from param vector
        #return self.I1 + jnp.exp(1j*rotangle)*self.P1
        return 1/jnp.sqrt(2) * self.H1

class CPGate(Gate): 
    def __init__(self, qids, n_params=1, fn=lambda x:x):
        super().__init__(dim=2, qids=qids, n_params=n_params, fn=fn)
        self.I1 = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.P1 = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

    def gate(self, params):
        # get relevant params from param vector
        rotangle = self.process(self.extract(params))
        #return self.I1 + jnp.exp(1j*rotangle)*self.P1
        return self.I1 + jnp.sin(rotangle)*self.P1 + 1j *jnp.cos(rotangle)*self.P1
    
class DispGate(Gate):
    """
    An arbitrary cavity displacement. Natively, it is parametrized by two real parameters:
    a (the real part of the displacement) and b (the imaginary part). If fn is specified,
    it should return an unpackable object: a, b = fn(param_vec_slice)
    """
    
    def __init__(self, dim, qids, n_params=2, fn=lambda x:x):
        super().__init__(dim, qids=qids, n_params=n_params, fn=fn)
        # hard-code the creation and annihilation operators
        self.cr = np.zeros((self.dim, self.dim))
        self.an = np.zeros((self.dim, self.dim))
        for i in range(self.dim-1):
            self.cr[i+1, i] = np.sqrt(i+1)
            self.an[i, i+1] = np.sqrt(i+1)
    
    def gate(self, params):
        a, b = self.process(self.extract(params))
        alpha = a + b*1j
        astar = alpha.conjugate()
        return expm(alpha*self.cr - astar*self.an)
    
class CPMGate(Gate):    
    def __init__(self, dim, qids, n_params=0, fn=lambda x:x):
        super().__init__(dim, qids=qids, n_params=n_params, fn=fn)
        # hard-code the creation and annihilation operators
        #self.id = jnp.eye(self.dim)
        self.id1 = np.eye(self.dim)
        for i in range(self.dim//2, self.dim):
            if i % 2 == 1:
                self.id1[i, i] = -1

    def gate(self, params):
        return self.id1

class SNAPGate(Gate):
    """
    A selective number-dependent arbitrary phase gate. Natively, it is parametrized by
    d real parameters, where d is the dimension of the cavity. qids should be a length-2
    list of ints [qubit_id, cavity_id]. If fn is specified, it should return a 1D ndarray
    of length d: theta_vec = fn(param_vec_slice)
    """
    
    def __init__(self, dim, qids, n_params=None, fn=lambda x:x):
        if n_params is None:
            n_params = dim//2
        super().__init__(dim, qids, n_params=n_params, fn=fn)
    
    def gate(self, params):
        theta = self.process(self.extract(params))
        theta = jnp.array(theta)
        diag = jnp.exp(jnp.concatenate((1j*theta, -1j*theta)))
        return jnp.diag(diag)

class GateLayer(Gate):
    
    def __init__(self, gates, regInfo):
        # try to order gates by qudit to minimize permutation during gate calculation
        self.gates = sorted(gates, key=lambda x: x.qudit_ids[0])
        # get the order of qids so that we know how to permute if necessary
        qids = []
        for g in self.gates:
            qids += g.qudit_ids
        # make sure this layer doesn't have any overlapping gates
        assert len(set(qids)) == len(qids)
        
        self.regInfo = regInfo
        self.permuted = qids + [len(qids) + qid for qid in qids]
        super().__init__(self.regInfo.dim, qids)
    
    def gate(self, params):
        # start with identity scalar and tensor-product all the gates
        mat = 1
        for g in self.gates:
            g = g.gate(params)
            mat = jnp.kron(mat, g)
        # if we have interleaved gates (i.e. a gate acting on qudit #1 and #3 but not #2)
        # then we need to un-permute the indices
        if self.permuted != self.regInfo.unpermuted:
            # reshape 2D unitary into qudit indices
            mat = mat.reshape(self.regInfo.shape)
            mat = jnp.moveaxis(mat, self.permuted, self.regInfo.unpermuted)
            mat = mat.reshape((self.regInfo.dim, self.regInfo.dim))
        return mat
