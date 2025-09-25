import pennylane as qml


class concrete_QCNN:
    def __init__(self, input, weights, last_layer_weights, num_wires=6):
        self.weights = weights
        self.input = input
        self.last_layer_weights = last_layer_weights
        self.wires = list(range(num_wires))
        self.num_wires = num_wires
        self.layers = weights.shape[1]
        self.dev = qml.device("default.qubit", wires=num_wires)

    def __call__(self):
        fun = qml.QNode(self.circuit, self.dev)
        return fun() + self.bias

    def __repr__(self):
        return str(qml.draw(self.circuit, decimals=2)())

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_ansatz_op(weights, wires=4):
        tmp = concrete_QCNN(None, weights, 0)
        O = qml.matrix(tmp.ansatz, wire_order=list(range(wires - 1, -1, -1)))()
        return O

    def encoding(self):
        qml.AmplitudeEmbedding(features=self.input, wires=self.wires, pad_with=0.5)

    def convolutional_layer(self, weights, wires, skip_first_layer=True):
        """Adds a convolutional layer to a circuit.
        Args:
            weights (np.array): 1D array with 15 weights of the parametrized gates.
            wires (list[int]): Wires where the convolutional layer acts on.
            skip_first_layer (bool): Skips the first two U3 gates of a layer.
        """
        n_wires = len(wires)
        assert n_wires >= 3, "this circuit is too small!"

        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    if indx % 2 == 0 and not skip_first_layer:
                        qml.U3(*weights[:3], wires=[w])
                        qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                    qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                    qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[9:12], wires=[w])
                    qml.U3(*weights[12:], wires=[wires[indx + 1]])

    def pooling_layer(self, weights, wires):
        """Adds a pooling layer to a circuit.
        Args:
            weights (np.array): Array with the weights of the conditional U3 gate.
            wires (list[int]): List of wires to apply the pooling layer on.
        """
        n_wires = len(wires)
        assert len(wires) >= 2, "this circuit is too small!"

        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                m_outcome = qml.measure(w)
                qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])
                # # Versione deferred:
                # qml.ctrl(qml.U3, (w), control_values=(1))(*weights, wires=wires[indx - 1])

    def dense_layer(sef, weights, wires):
        """Apply an arbitrary unitary gate to a specified set of wires."""
        qml.ArbitraryUnitary(weights, wires)

    def ansatz(self):
        wires = self.wires.copy()

        for j in range(self.layers):
            """Apply both the convolutional and pooling layer."""
            self.convolutional_layer(self.weights[:15], wires, skip_first_layer=(not j == 0))
            self.pooling_layer(self.weights[15:], wires)

            wires = wires[::2]
            qml.Barrier(wires=wires, only_visual=True)

        self.dense_layer(self.last_layer_weights, wires)


    def circuit(self):
        self.encoding()
        qml.Barrier(wires=list(range(self.num_wires)), only_visual=True)
        self.ansatz()

        return qml.probs(wires=(0))


