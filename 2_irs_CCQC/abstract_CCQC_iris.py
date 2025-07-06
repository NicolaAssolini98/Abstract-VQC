import pennylane as qml
from intervalVQC import intervalVQC
from concrete_CCQC_iris import concrete_CCQC

# dev = qml.device("default.qubit", wires=4)
class abstract_CCQC:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.ab_circuit = intervalVQC(2, use_clip=True)

    def __call__(self, data):
        # encoding
        self.ab_circuit.Ry(0, data[0])
        self.ab_circuit.CNOT(0, 1)
        self.ab_circuit.Ry(1, data[1])
        self.ab_circuit.CNOT(0, 1)
        self.ab_circuit.Ry(1, data[2])
        self.ab_circuit.PauliX(0)
        self.ab_circuit.CNOT(0,1)
        self.ab_circuit.Ry(1, data[3])
        self.ab_circuit.CNOT(0,1)
        self.ab_circuit.Ry(1, data[4])
        self.ab_circuit.PauliX(0)

        # ansatz
        # concrete = concrete_CCQC(None, self.weights, None)
        ansatz_op = concrete_CCQC.get_ansatz_op(self.weights)
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

        return prob_0, prob_1 - self.bias  # We classify as 0 if prob_0 > prob_1 - bias, otherwise as 1

