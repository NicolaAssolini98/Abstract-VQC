import pennylane as qml
from intervalVQC import intervalVQC
from concrete_CCQC_digits import concrete_CCQC

# dev = qml.device("default.qubit", wires=4)
class abstract_CCQC:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.ab_circuit = intervalVQC(8, use_clip=True)

    def __call__(self, data):
        # encoding
        for i in range(len(data[0])):
            print(f"{i}/{len(data[0])-1}")
            self.ab_circuit.hadamard(i)
            # Apply angle embeddings based on the feature values
        for i in range(len(data)):
            print(f"{i}/{len(data)-1}")
            # For odd-indexed features, use Z-rotation in the angle embedding
            if i % 2:
                for j in range(len(data[i])):
                    # Apply angle embedding for each feature value
                    self.ab_circuit.Rz(j, data[i][j])
            # For even-indexed features, use X-rotation in the angle embedding
            else:
                for j in range(len(data[i])):
                    # Apply angle embedding for each feature value
                    self.ab_circuit.Rx(j, data[i][j])
        print("Abstract circuit encoding done")
        # ansatz
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

        return prob_0, prob_1 - self.bias


