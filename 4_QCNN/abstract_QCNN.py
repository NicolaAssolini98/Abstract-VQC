import numpy as np
from interval import interval

from complex_interval import ComplexInterval
from intervalVQC import intervalVQC
from concrete_QCNN import concrete_QCNN

# dev = qml.device("default.qubit", wires=4)
class abstract_QCNN:
    def __init__(self, weights, last_layer_weights, num_wires=6):
        self.weights = weights
        self.last_layer_weights = last_layer_weights
        self.num_wires = num_wires

    def __call__(self, data):
        ab_circuit = intervalVQC(self.num_wires, use_clip=True)

        # Encoding
        # TODO
        ab_circuit.state = np.array([[ComplexInterval(data[i], interval([0, 0]))] for i in range(len(data))])

        ansatz_op = concrete_QCNN.get_ansatz_op(weights=self.weights, last_layer_weights=self.last_layer_weights)
        ab_circuit.execute_operator(ansatz_op)

        # Measurement
        result = ab_circuit.get_measurement_interval()

        prob_0 = 0
        prob_1 = 0

        for i in range(len(result)):
            if i % 2 == 0:
                prob_0 += result[i]
            else:
                prob_1 += result[i]

        return prob_0, prob_1 # We classify as 0 if prob_0 > prob_1 - bias, otherwise as 1


