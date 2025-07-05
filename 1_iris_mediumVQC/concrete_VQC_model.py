
import pennylane as qml
from pennylane import numpy as np
from interval import interval

# dev = qml.device("default.qubit", wires=4)
class concrete_VQC:
    def __init__(self, data, weights):
        assert len(data) == 4, "Input data must have 4 features."
        assert len(weights) == 4, "Weights must have 4 elements."
        self.weights = weights
        self.input = data
        self.dev = qml.device("default.qubit", wires=4)

    def __call__(self):
        fun = qml.QNode(self.circuit, self.dev)
        return fun()

    def __repr__(self):
        return str(qml.draw(self.circuit, decimals=2)())

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_ansatz_op(weights):
        tmp = concrete_VQC([1,1,1,1], weights)
        O = qml.matrix(tmp.ansatz, wire_order=[3, 2, 1, 0])()
        return O #np.asarray(O)

    def encoding(self):
        for i in range(len(self.input)):
            qml.RX(self.input[i], wires=i)
        # qml.RX(self.input[1], wires=1)
        # qml.RX(self.input[2], wires=2)
        # qml.RX(self.input[3], wires=3)

    def ansatz(self):
        for i in range(len(self.weights)-1):
            qml.CNOT(wires=[i, (i + 1)])
        qml.CNOT(wires=[3, 0])
        for i in range(len(self.weights)):
            qml.RY(self.weights[i], wires=i)

        # qml.CNOT(wires=[0, 1])
        # qml.CNOT(wires=[1, 2])
        # qml.CNOT(wires=[2, 3])
        # qml.CNOT(wires=[3, 0])
        # qml.RY(self.weights[0], wires=0)
        # qml.RY(self.weights[1], wires=1)
        # qml.RY(self.weights[2], wires=2)
        # qml.RY(self.weights[3], wires=3)

    def circuit(self):
        self.encoding()
        self.ansatz()
        return qml.expval(qml.PauliZ(0))




'''

P(0) - P(1) = N = vqc()
P(0) + P(1) = 1 -> P(0) = 1 - P(1)


1 - P(1) - P(1) = N -> 1 - 2*P(1) = N -> P(1) = (1 - N)/2
'''
