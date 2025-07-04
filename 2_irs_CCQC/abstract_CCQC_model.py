import pennylane as qml
from intervalVQC import intervalVQC
from concrete_CCQC_model import concrete_CCQC

# dev = qml.device("default.qubit", wires=4)
class abstract_CCQC:
    def __init__(self, weights, bias, layers):
        self.weights = weights
        self.bias = bias
        self.layer = layers
        self.ab_circuit = intervalVQC(2, use_clip=True)

    def __call__(self, data):
        # encoding
        self.ab_circuit.Ry(data[0], 0)
        self.ab_circuit.CNOT(0, 1)
        self.ab_circuit.Ry(data[1], 1)
        self.ab_circuit.CNOT(0, 1)
        self.ab_circuit.Ry(data[2], 1)
        self.ab_circuit.PauliX(0)
        self.ab_circuit.CNOT(0,1)
        self.ab_circuit.Ry(data[3], 1)
        self.ab_circuit.CNOT(0,1)
        self.ab_circuit.Ry(data[4], 1)
        self.ab_circuit.PauliX(0)

        # ansatz
        concrete = concrete_CCQC(None, self.weights, None)
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


