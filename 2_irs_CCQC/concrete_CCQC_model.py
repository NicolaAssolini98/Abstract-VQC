
import pennylane as qml
from pennylane import numpy as np

# dev = qml.device("default.qubit", wires=4)
class concrete_CCQC:
    def __init__(self, data, weights, bias):
        assert len(data) == 5, "Input data must have 5 features."
        self.weights = weights
        self.input = data
        self.bias = bias
        self.dev = qml.device("default.qubit", wires=2)

    def __call__(self):
        fun = qml.QNode(self.circuit, self.dev)
        return fun() + self.bias

    def __repr__(self):
        return str(qml.draw(self.circuit, decimals=2)())

    def __str__(self):
        return self.__repr__()

    def get_ansatz_op(self):
        return qml.matrix(self.ansatz, wire_order=[0, 1])()

    def encoding(self):
        qml.RY(self.input[0], wires=0)
        qml.CNOT(wires=[0, 1])

        qml.RY(self.input[1], wires=1)
        qml.CNOT(wires=[0, 1])

        qml.RY(self.input[2], wires=1)
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])

        qml.RY(self.input[3], wires=1)
        qml.CNOT(wires=[0, 1])

        qml.RY(self.input[4], wires=1)
        qml.PauliX(wires=0)

    def ansatz(self):
        for layer_weights in self.weights:
            for wire in range(2):
                qml.Rot(*layer_weights[wire], wires=wire)
            qml.CNOT(wires=[0, 1])

    def circuit(self):
        self.encoding()
        self.ansatz()

        return qml.expval(qml.PauliZ(0))


