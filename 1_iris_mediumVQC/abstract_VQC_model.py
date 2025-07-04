
import pennylane as qml
from pennylane import numpy as np
from interval import interval
from intervalVQC import intervalVQC
from concrete_VQC_model import concrete_VQC


class abstract_VQC:
    def __init__(self, weights):
        assert len(weights) == 4, "Weights must have 4 elements."
        self.weights = weights
        self.ab_circuit = intervalVQC(4, use_clip=True)
        # self.dev = qml.device("default.qubit", wires=4)

    def __call__(self, data):
        assert len(data) == 4, "Input data must have 4 features."
        # encoding
        for i in range(len(data)):
            self.ab_circuit.Rx(i, data[i])
        # ansatz
        concrete = concrete_VQC([1,1,1,1], self.weights)
        ansatz_op = concrete.get_ansatz_op()
        self.ab_circuit.execute_operator(ansatz_op)
        # measurement
        result = self.ab_circuit.get_measurement_interval()

        prob_0 = 0
        prob_1 = 0

        for i in range(len(result)):
            if i % 2 == 0:
                prob_0 += result[i]
            else:
                prob_1 += result[i]

        return (prob_0, prob_1)

    # def __repr__(self):
    #     return str(qml.draw(self.circuit, decimals=2)())
    #
    # def __str__(self):
    #     return self.__repr__()


'''
def encoding(a):
    qml.RX(a[0], wires=0)
    qml.RX(a[1], wires=1)
    qml.RX(a[2], wires=2)
    qml.RX(a[3], wires=3)

def ansatz(weights):
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)

@qml.qnode(dev)
def vqc(weights, a):
    encoding(a)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))


def run(weights, a):
    return vqc(weights, a)

def circuit_string(weights, a):
    return qml.draw(vqc)(weights, a)
# print(vqc([0,1,2,3],[1,1,1,1]))
'''