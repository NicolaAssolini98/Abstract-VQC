
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
        # concrete = concrete_VQC([1,1,1,1], self.weights)
        ansatz_op = concrete_VQC.get_ansatz_op(self.weights)
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

        return prob_0, prob_1

