import pennylane as qml


class concrete_CCQC:
    def __init__(self, data, weights, bias):
        self.weights = weights
        self.input = data
        self.bias = bias
        self.num_wires = len(self.input[0]) if data is not None else 4
        self.dev = qml.device("default.qubit", wires=self.num_wires)

    def __call__(self):
        fun = qml.QNode(self.circuit, self.dev)
        return fun() + self.bias

    def __repr__(self):
        return str(qml.draw(self.circuit, decimals=2)())

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_ansatz_op(weights, wires=4):
        tmp = concrete_CCQC(None, weights, 0)
        O = qml.matrix(tmp.ansatz, wire_order=list(range(wires - 1, -1, -1)))()
        return O

    def encoding(self):
        # Apply Hadamard gates to all qubits to create an equal superposition state
        for i in range(len(self.input[0])):
            qml.Hadamard(i)

        # Apply angle embeddings based on the feature values
        for i in range(len(self.input)):
            # For odd-indexed features, use Z-rotation in the angle embedding
            if i % 2:
                qml.AngleEmbedding(features=self.input[i], wires=range(self.num_wires), rotation="Z")
            # For even-indexed features, use X-rotation in the angle embedding
            else:
                qml.AngleEmbedding(features=self.input[i], wires=range(self.num_wires), rotation="X")

        # Define the ansatz (quantum circuit ansatz) for parameterized quantum operations

    def ansatz(self):
        # Apply RY rotations with the first set of parameters
        for i in range(self.num_wires):
            qml.RY(self.weights[i], wires=i)

        # Apply CNOT gates with adjacent qubits (cyclically connected) to create entanglement
        for i in range(self.num_wires):
            qml.CNOT(wires=[(i - 1) % self.num_wires, (i) % self.num_wires])

        # Apply RY rotations with the second set of parameters
        for i in range(self.num_wires):
            qml.RY(self.weights[i + self.num_wires], wires=i)

        # Apply CNOT gates with qubits in reverse order (cyclically connected)
        # to create additional entanglement
        for i in range(self.num_wires):
            qml.CNOT(wires=[(self.num_wires - 2 - i) % self.num_wires, (self.num_wires - i - 1) % self.num_wires])

    def circuit(self):
        self.encoding()
        self.ansatz()

        return qml.expval(qml.PauliZ(0))


