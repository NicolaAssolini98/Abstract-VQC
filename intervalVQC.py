import numpy as np
from complex_interval import ComplexInterval
import pennylane as qml
# from qiskit import *
# from qiskit.quantum_info import *
from interval import *
np.set_printoptions(precision=3, suppress=True)



class intervalVQC:
    def __init__(self, n_qubit, use_clip=True):

        self.n_qubit = n_qubit
        self.use_clip = use_clip
        self.state = np.array([[ComplexInterval(interval([0, 0]), interval([0, 0]))] for _ in range(2**self.n_qubit)])
        self.state[0][0] = ComplexInterval(interval([1, 1]), interval([0, 0]))

    def zero_int(self):
        return ComplexInterval(interval([0, 0]), interval([0, 0]))

    def hadamard(self, n):
        """
        Get the Hadamard gate for the n-th qubit in a quantum circuit with tot_q qubits.
        :param n: The qubit index.
        :param tot_q: The total number of qubits in the quantum circuit.
        :return: The Hadamard gate for the n-th qubit.
        """
        O = qml.matrix(lambda: qml.Hadamard(wires=n), wire_order=list(range(self.n_qubit - 1, -1, -1)))()
        self.state = O @ self.state
        if self.use_clip:
            for i in self.state:
                i[0].clip()


    def PauliX(self, n):
        """
        Get the Pauli-X gate for the n-th qubit in a quantum circuit with tot_q qubits.
        :param n: The qubit index.
        :param tot_q: The total number of qubits in the quantum circuit.
        :return: The Pauli-X gate for the n-th qubit.
        """
        O = qml.matrix(lambda: qml.PauliX(wires=n), wire_order=list(range(self.n_qubit - 1, -1, -1)))()
        self.state = O @ self.state
        if self.use_clip:
            for i in self.state:
                i[0].clip()


    def CNOT(self, n, m):
        """
        Get the
        :param n:  Controller index.
        :param m:  The qubit index.
        :param tot_q: The total number of qubits in the quantum circuit.
        :return: The CNOT gate for the n-th and m-th qubits.
        """
        O = qml.matrix(lambda: qml.CNOT(wires=[n,m]), wire_order=list(range(self.n_qubit - 1, -1, -1)))()
        self.state = O @ self.state
        # qc = QuantumCircuit(self.n_qubit)
        # qc.cx(n, m)
        # self.state = Operator(qc).data @ self.state
        if self.use_clip:
            for i in self.state:
                i[0].clip()


    def Rz(self, n, theta_int):
        assert n < self.n_qubit
        r = np.array([[ComplexInterval(imath.cos(theta_int / 2), -(imath.sin(theta_int / 2))), self.zero_int()],
                    [self.zero_int(), ComplexInterval(imath.cos(theta_int / 2), imath.sin(theta_int / 2))]])
        self.state = (np.kron(np.kron(np.identity(2**(self.n_qubit - 1 - n), dtype=complex), r), np.identity(2 ** n, dtype=complex))) @ self.state
        if self.use_clip:
            for i in self.state:
                i[0].clip()


    def Rx(self, n, theta_int):
        assert n < self.n_qubit
        r = np.array([[ ComplexInterval(imath.cos(theta_int / 2), interval([0,0])), ComplexInterval(interval([0,0]), -imath.sin(theta_int / 2)) ],
                    [ ComplexInterval(interval([0,0]), -imath.sin(theta_int / 2)), ComplexInterval(imath.cos(theta_int / 2), interval([0,0])) ]])
        self.state = (np.kron(np.kron(np.identity(2**(self.n_qubit - 1 - n), dtype=complex), r), np.identity(2 ** n, dtype=complex))) @ self.state
        if self.use_clip:
            for i in self.state:
                i[0].clip()


    def Ry(self, n, theta_int):
        assert n < self.n_qubit
        r = np.array([[ ComplexInterval(imath.cos(theta_int / 2), interval([0,0])), ComplexInterval(-imath.sin(theta_int / 2), interval([0,0]))],
                    [ ComplexInterval(imath.sin(theta_int / 2), interval([0,0])), ComplexInterval(imath.cos(theta_int / 2), interval([0,0])) ]])
        self.state = (np.kron(np.kron(np.identity(2**(self.n_qubit - 1 - n), dtype=complex), r), np.identity(2 ** n, dtype=complex))) @ self.state
        if self.use_clip:
            for i in self.state:
                i[0].clip()


    def execute_operator(self, operator):
        """
        Execute a given operator on the current state.
        :param operator: The operator to be applied.
        """
        self.state = operator @ self.state
        if self.use_clip:
            for i in self.state:
                i[0].clip()


    def get_measurement_interval(self): 
        result = []
        if self.use_clip:
            for el in self.state:
                result.append((el[0].abs_powered() & interval([0.0, 1.0])) *100)
        else:
            for el in self.state:
                result.append((el[0].abs_powered()))
        
        return result


    def __repr__(self):
        print()
        for i in range(2**self.n_qubit):
            if i < 10:
               print(f" {i}: {round(self.state[i][0],3)}")
            else:
                print(f"{i}: {round(self.state[i][0], 3)}")
                
        return ""
    