from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


def str2idx(p): # II, IX, IY, IZ, XI, XX...
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
    s = fid[0]
    t = fid[0]
    if fid[1] == '0':
        s = 'I' + s
    else:
        s = 'Z' + s
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

