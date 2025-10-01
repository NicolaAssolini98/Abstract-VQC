import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import seaborn as sns
import jax
import jax.numpy as jnp
import optax  # optimization using jax

import pennylane as qml
import pennylane.numpy as pnp

import warnings
from interval import interval

from complex_interval import ComplexInterval
from concrete_QCNN import concrete_QCNN
from abstract_QCNN import abstract_QCNN
from intervalVQC import intervalVQC

warnings.filterwarnings("ignore")
# np.random.seed(42)

print_state = False

def ansatz():
    qml.CNOT(wires=[0,1])


def circuit_probs(inp, wires):
    qml.AmplitudeEmbedding(features=inp, wires=wires)
    ansatz()
    # print(qml.state())
    return qml.probs(0)


def circuit_state(inp, wires):
    qml.AmplitudeEmbedding(features=inp, wires=wires)
    ansatz()
    # print(qml.state())
    return qml.state()


def concrete(inp, wires):
    dev = qml.device("default.qubit", wires=len(wires))
    print(qml.QNode(circuit_probs, dev)(inp, wires))
    if print_state:
        print(qml.QNode(circuit_state, dev)(inp, wires))


def abstract(inp, wires):
    ab_circuit = intervalVQC(len(wires), use_clip=True)
    state = np.array(inp)
    n = int(np.log2(len(state)))
    indices = np.arange(len(state))
    reversed_indices = [int(f"{i:0{n}b}"[::-1], 2) for i in indices]
    rst = state[reversed_indices]
    ab_circuit.state = np.array([[ComplexInterval(interval(rst[i][0][0], rst[i][0][1]), interval([0, 0]))] for i in range(len(rst))])
    if print_state:
        print(ab_circuit.state.flatten())

    O = qml.matrix(ansatz, wire_order=list(range(len(wires) - 1, -1, -1)))()
    ab_circuit.execute_operator(O)

    result = ab_circuit.get_measurement_interval()
    # print(result)
    prob_0 = 0
    prob_1 = 0

    for i in range(len(result)):
        if i % 2 == 0:
            prob_0 += result[i]
        else:
            prob_1 += result[i]

    print(prob_0, prob_1)


def main_local():
    n_wires = 6
    wires = list(range(n_wires))

    feats = np.random.rand(2 ** n_wires)
    feats = feats / np.linalg.norm(feats)

    concrete(feats, wires)
    print('----------')

    eps = 0.0
    interval_feats = [interval([x - eps, x + eps]) for x in feats]
    # print(interval_feats)

    abstract(interval_feats, wires)

def main_global():
    n_wires = 6
    for _ in range(5):
        w, lw, feat = np.random.rand(18, 2), np.random.rand(4 ** 2 - 1), np.random.rand(2 ** n_wires)
        feat = feat / np.linalg.norm(feat)
        qcnn = concrete_QCNN(weights=w, last_layer_weights=lw, input=feat, num_wires=n_wires)
        aqcnn = abstract_QCNN(weights=w, last_layer_weights=lw, num_wires=n_wires)

        prediction = qcnn()
        # print("-> ",(1-(1-vqc())/2)*100,((1-vqc())/2)*100)
        print("C -> ", prediction)
        eps = 0.1
        p0, p1 = aqcnn([interval([x - eps, x + eps]) for x in feat])
        print("A -> ", p0, p1)
        print('---')

if __name__ == "__main__":
    main_global()


