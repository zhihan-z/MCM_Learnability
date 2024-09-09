"""Some utility functions."""
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


def str2idx(p): # II, IX, IY, IZ, XI, XX...
    """From a Pauli string to its index in the Pauli transfer matrix.
    That is, we turn II, IX, IY, IZ, XI, XX... to 0, 1, 2, 3, 4, 5...
    @param p The Pauli string.
    """
    ret = 0
    for i in range(2):
        if p[i] == "X":
            ret += 1 * 4 ** (1 - i)
        elif p[i] == "Y":
            ret += 2 * 4 ** (1 - i)
        elif p[i] == "Z":
            ret += 3 * 4 ** (1 - i)
    return ret


def fid2st(fid):
    """From an Pauli fidelity to its starting Pauli string and its ending Pauli string (and the sign).
    Specifically, the input Qxy represents lambda^Q_{xy}.
    The starting Pauli string is G^dagger(Q otimes Z^x) and the ending is Q otimes Z^y.
    Note that qiskit uses a reverse ordering of the qubits.
    @param fid The Pauli fidelity.
    """
    s = fid[0]
    t = fid[0]
    # Q otimes Z^x
    if fid[1] == '0':
        s = 'I' + s
    else:
        s = 'Z' + s
    # Q otimes Z^y
    if fid[2] == '0':
        t = 'I' + t
    else:
        t = 'Z' + t
    cn_circ = QuantumCircuit(2, 0)
    cn_circ.cnot(0, 1)
    s = Pauli(s).evolve(cn_circ, frame='h').to_label()
    sgn = 1
    if s[0] == '-':
        sgn *= -1
        s = s[1:]
    return sgn, s, t

