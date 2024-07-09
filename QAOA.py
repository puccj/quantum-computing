import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator, Statevector
from qiskit.primitives import Estimator
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt


class MaxCutSolver:
    def __init__(self, graph, p=1):
        self.graph = graph
        self.p = p
        self.n = len(graph)
        self.qc = None
        self.operator = self.__get_operator()

    def set_p(self, p):
        self.p = p
        self.qc = None

    def draw_graph(self, solution = None, circular_layout=False):
        if solution is not None:
            if len(solution) != len(self.graph):
                raise ValueError('The solution must have the same length as the graph')
            # n_c = ['blue' if solution[i] == 0 else 'green' for i in range(len(solution))]
            color_map = []
            for node in self.graph:
                if solution[node] == 0:
                    color_map.append('lightblue')
                else: 
                    color_map.append('green')
            
            edge_color = ['gray'] * self.graph.number_of_edges()
            for i, edge in enumerate(self.graph.edges()):
                if solution[edge[0]] != solution[edge[1]]:
                    edge_color[i] = 'red'
                    
        else:
            color_map = 'lightblue'
            edge_color = 'gray'
        
        if (circular_layout):
            pos = nx.circular_layout(self.graph, seed=42)
        else:
            pos = nx.spring_layout(self.graph, seed=42)

        nx.draw(self.graph, pos, node_color=color_map, edge_color=edge_color, node_size=100, with_labels=True)

    
    def get_adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph).toarray()
    
    def __get_operator(self):
        pauli_list = []     # will store the Pauli operators
        coeffs = []         # and their corresponing coefficients
        shift=0

        for i,j in self.graph.edges():
            if self.graph[i][j] != 0: 
                # for each non-zero entry create a Pauli operator Z_i Z_j
                x_p = np.zeros(self.n, dtype=bool)   # all false (no X operators)
                z_p = np.zeros(self.n, dtype=bool)
                
                # Z operators on the corresponding qubits
                z_p[i] = True
                z_p[j] = True

                pauli_list.append(Pauli((z_p, x_p)))
                coeffs.append(0.5)
                shift -= 0.5
        
        # we need to adjust the Hamiltonian for the overall offset
        h1=SparsePauliOp.from_operator(Operator(shift*np.identity(2**self.n)))
        h2=SparsePauliOp(pauli_list, coeffs=coeffs)

        return SparsePauliOp.sum([h1,h2])
    
    def get_cost(self, string):
        'Receives a string of 0 and 1s and gives back its cost to the MaxCut hamiltonian'
        configuration = [int(i) for i in string]

        configuration = [1-x*2 for x in configuration]
        cost = 0
        for i,edge in enumerate(self.graph.edges):
            cost += (configuration[edge[0]]*configuration[edge[1]])
        return cost
    
    def __sample_most_likely(self, state_vector):
        """Compute the most likely bitstring to come out in a measurement from state vector."""
        values = state_vector
        num_qubits = int(np.log2(len(values)))
        
        #index of element in state_vector with largest magnitude -> highest probability
        k = np.argmax(np.abs(values))

        x = [int(digit) for digit in np.binary_repr(k, num_qubits)]     # bitfield
        
        # the convention in QC is to put the least significant bit at start of string, so:
        x.reverse() 
        return np.asarray(x)

    def classical_solution(self):
        npme = NumPyMinimumEigensolver()
        result = npme.compute_minimum_eigenvalue(Operator(self.operator))
        return self.__sample_most_likely(result.eigenstate.data)
    
    def create_qaoa_circuit(self, p = None):
        """Create the parametrized quantum circuit for the QAOA algorithm."""

        if p is not None:
            self.p = p

        gamma=ParameterVector('gamma',self.p)
        beta=ParameterVector('beta',self.p)
        qr=QuantumRegister(self.n, 'q')
        qc=QuantumCircuit(qr)
        qc.h(qr[0:self.n])   #initialize the qubits in superposition
        
        #repeat the application of U(C,gamma) and U(B,beta) p times
        for i in range(self.p):
            for edge in self.graph.edges():
                qc.cx(edge[0],edge[1])
                qc.rz(2*gamma[i],edge[1])
                qc.cx(edge[0],edge[1])
            qc.barrier()

            qc.rx(2*beta[i],qr[0:self.n]) #mixer layer
            qc.barrier()
        
        self.qc = qc
        return qc
    
    def draw_qaoa_circuit(self):
        if self.qc is None:
            self.create_qaoa_circuit()
        return self.qc.draw('mpl')
    
    def optimize_and_plot(self, p = None):
        if p is not None:
            self.set_p(p)
        
        if (self.qc is None):
            self.create_qaoa_circuit()
        
        optimizer=partial(minimize,method='L-BFGS-B')
        estimator=Estimator()
        start=np.full(2*self.p,0.2)
        sampling_vqe = VQE(estimator, self.qc, optimizer=optimizer, initial_point=start)
        result = sampling_vqe.compute_minimum_eigenvalue(self.operator)
        qc=result.optimal_circuit.assign_parameters(result.optimal_point)
        state = Statevector(qc).data

        plt.figure(self.p, figsize=(20,6))
        xticks = range(0, 2**len(self.graph))
        xtick_labels = list(map(lambda x: format(x, "0{}b".format(self.n)), xticks))
        plt.title("p={}".format(self.p))
        plt.xlabel("bitstrings")
        plt.ylabel("freq.")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.bar(xticks, (state*np.conjugate(state)).real) # plot real part of the amplitudes squared (probabilities)

        return state