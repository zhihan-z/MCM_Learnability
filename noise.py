from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit.quantum_info import Operator, PTM, Pauli, random_quantum_channel
import numpy as np
from collections import defaultdict

from utils import str2idx, fid2st

class Noise:
    def __init__(self, noise_scale):
        self._id_gate = Operator(np.eye(4))
        self._id_gate3 = Operator(np.eye(8))
        self._random_qi(noise_scale)
        self._random_refresh(noise_scale * 0.9) # To make random qi and random measure-and-prepare have similar error rates.
        self._random_spam(noise_scale * 0.25) # Terminating measurement error rates should be smaller than that of MCM's.
        self.noise_model = NoiseModel()
        self.noise_model.add_basis_gates(['unitary'])
        self.noise_model.add_all_qubit_quantum_error(self._err_pre, 'pre')
        self.noise_model.add_all_qubit_quantum_error(self._err_post, 'post')
        self.noise_model.add_all_qubit_quantum_error(self._err_spam, 'spam')
        self.noise_model.add_all_qubit_quantum_error(self._err_qi, 'qi')
        self.backend_noisy = AerSimulator(noise_model=self.noise_model)
    

    def noisy_mcm(self, circ, cbit):
        """Noisy mid-circuit measurement.
        The measurement is in the measure-and-prepare fashion:
        We apply a Pauli noise, then an ideal mid-circuit measurement, followed by another Pauli noise.
        """
        circ.cnot(0, 1)
        circ.unitary(self._id_gate, [0, 1], label='pre')
        circ.measure(1, cbit)
        circ.unitary(self._id_gate, [0, 1], label='post')
    

    def compiled_mcm(self, circ, cbit):
        """Noisy mid-circuit measurement.
        The noise is a general quantum instrument and we perform randomized compiling on it.
        """
        p = np.random.choice(['I', 'X', 'Y', 'Z'])
        alpha, beta, gamma = np.random.randint(2, size=3)
        if alpha == 0 and beta == 0:
            p12 = 'I' + p
        elif alpha == 0 and beta == 1:
            p12 = 'Z' + p
        elif alpha == 1 and beta == 0:
            p12 = 'X' + p
        else:
            p12 = 'Y' + p
        cn_circ = QuantumCircuit(2, 0)
        cn_circ.cnot(0, 1)
        p12 = Pauli(p12).evolve(cn_circ, frame='h').to_label()
        if p12[0] == '-':
            p12 = p12[1:]
        for i in range(2):
            if p12[i] == 'X':
                circ.x(1 - i)
            elif p12[i] == 'Y':
                circ.y(1 - i)
            elif p12[i] == 'Z':
                circ.z(1 - i)
        circ.reset(2)
        circ.cnot(0, 1)
        circ.unitary(self._id_gate3, [0, 1, 2], label='qi')
        if alpha == 1:
            circ.x(2)
        circ.measure(2, cbit)
        if p == 'X':
            circ.x(0)
        elif p == 'Y':
            circ.y(0)
        elif p == 'Z':
            circ.z(0)
        if gamma == 1:
            circ.z(1)
        if alpha == 1:
            circ.x(1)

        
    def noisy_sp(self, circ, p):
        """Noisy state preparation.
        Though almost any initial state works for our protocol, it would be better for the initial state to have big overlap with the ideal operator.
        Here we assume that we prepare the eigenstate of the ideal operator and then apply a special Pauli noise (same as the terminating measurement nosie).
        """
        p = p[::-1]
        for i in range(2):
            if p[i] == 'I':
                p = p[:i] + np.random.choice(['X', 'Y', 'Z']) + p[i + 1:]
            if p[i] == 'X':
                circ.h(i)
            elif p[i] == 'Y':
                circ.h(i)
                circ.s(i)
        circ.unitary(self._id_gate, [0, 1], label='spam')
    

    def noisy_m(self, circ, p, cbit):
        """Noisy terminating measurement.
        We apply a special Pauli noise and then make an ideal terminating measurement.
        If the measured operator is I, then we will still make a measurement but will ignore the output.
        """
        p = p[::-1]
        circ.unitary(self._id_gate, [0, 1], label='spam')
        for i in range(2):
            if p[i] == 'I':
                p = p[:i] + np.random.choice(['X', 'Y', 'Z']) + p[i + 1:]
            if p[i] == 'X':
                circ.h(i)
            elif p[i] == 'Y':
                circ.sdg(i)
                circ.h(i)
        circ.measure([0, 1], [cbit, cbit + 1])
    

    def _random_qi(self, noise_scale):
        """Generate a quantum instrument for noisy subsystem measurement.
        It is hard to satisfy complete positivity if we construct arbitrarily.
        So we have to rely on some physical structures.
        We compose noise before measurement + misreport error + noise after measurement, then average several instances to form a joint probability distribution.
        The helper channel is Q_0\otimes I+Q_1\otimes X.
        """
        tot_q0 = np.zeros([16, 16], dtype=complex)
        tot_q1 = np.zeros([16, 16], dtype=complex)
        tot_weight = 0.0
        for _ in range(2):
            p01 = np.random.rand() * noise_scale
            p10 = np.random.rand() * noise_scale
            q0 = np.kron(np.array([[1-p01+p10,0,0,1-p01-p10],[0,0,0,0],[0,0,0,0],[1-p01-p10,0,0,1-p01+p10]])/2, np.eye(4))
            q1 = np.kron(np.array([[1-p10+p01,0,0,-1+p10+p01],[0,0,0,0],[0,0,0,0],[-1+p10+p01,0,0,1-p10+p01]])/2, np.eye(4))
            n00 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            n01 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            n10 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            n11 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            weight = np.random.rand()
            tot_weight += weight
            tot_q0 += np.kron(n00, n01) @ q0 @ np.kron(n10, n11) * weight
            tot_q1 += np.kron(n00, n01) @ q1 @ np.kron(n10, n11) * weight
        q0 = np.kron(np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2, np.eye(4))
        q1 = np.kron(np.array([[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]])/2, np.eye(4))
        e = ['X', 'Y']
        for i in range(2):
            rate0 = {}
            rate0['X' + e[i]] = noise_scale * (4 + np.random.rand())
            rate0['II'] = 1 - rate0['X' + e[i]]
            n0 = PTM(pauli_error(list(rate0.items()))).data
            rate1 = {}
            rate1['XI'] = noise_scale * 4 * i
            rate1['II'] = 1 - rate1['XI']
            n1 = PTM(pauli_error(list(rate1.items()))).data
            weight = np.random.rand() + 1
            tot_weight += weight
            tot_q0 += n0 @ q0 @ n1 * weight
            tot_q1 += n0 @ q1 @ n1 * weight
        tot_q0 /= tot_weight
        tot_q1 /= tot_weight
        helper_chan = PTM(np.kron(np.eye(4), tot_q0) + np.kron(np.diag([1,1,-1,-1]), tot_q1))
        assert helper_chan.is_cptp()
        # now we have the instrument, calculate the error rates and fidelities
        rate = {}
        fid = defaultdict(float)
        channels = [tot_q0, tot_q1]
        p2 = ['I', 'Z']
        p4 = ['I', 'X', 'Y', 'Z']
        for a in range(2):
            for b in range(2):
                Lab = np.zeros(4)
                for x in range(2):
                    for A in range(2):
                        for B in range(4):
                            for C in range(2):
                                Lab[B] += np.real_if_close((-1) ** ((b + x) * A + (a + x) * C) * channels[x][str2idx(p2[A] + p4[B]), str2idx(p2[C] + p4[B])]) / 4
                rate['I' + str(a) + str(b)] = (Lab[0] + Lab[1] + Lab[2] + Lab[3]) / 4
                rate['X' + str(a) + str(b)] = (Lab[0] + Lab[1] - Lab[2] - Lab[3]) / 4
                rate['Y' + str(a) + str(b)] = (Lab[0] - Lab[1] + Lab[2] - Lab[3]) / 4
                rate['Z' + str(a) + str(b)] = (Lab[0] - Lab[1] - Lab[2] + Lab[3]) / 4
                for B in range(4):
                    for c in range(2):
                        for d in range(2):
                            fid[p4[B] + str(c) + str(d)] += Lab[B] * (-1) ** (a * c + b * d)

        self._err_qi = helper_chan
        self.fid_qi = fid
        self.rate_qi = rate
        return
    

    def _random_refresh(self, noise_scale):
        """Noises for measure-and-prepare mid-circuit measurement. Noises are Pauli channels.
        """
        rate_pre = {}
        for i in ["I", "X", "Y", "Z"]:
            for j in ["I", "X", "Y", "Z"]:
                rate_pre[i+j] = np.random.rand() * noise_scale ** ((i != 'I') + (j != 'I'))
        rate_pre["II"] += 1 - np.sum([v for v in rate_pre.values()])
        err_pre = pauli_error(list(rate_pre.items()))
        fid_pre = np.real_if_close(np.diag(PTM(err_pre).data))

        rate_post = {}
        for i in ["I", "X", "Y", "Z"]:
            for j in ["I", "X", "Y", "Z"]:
                rate_post[i+j] = np.random.rand() * noise_scale ** ((i != 'I') + (j != 'I'))
        rate_post["II"] += 1 - np.sum([v for v in rate_post.values()])
        err_post = pauli_error(list(rate_post.items()))
        fid_post = np.real_if_close(np.diag(PTM(err_post).data))

        fid_refresh = {}
        for i in ["I", "X", "Y", "Z"]:
            for j in ["0", "1"]:
                for k in ["0", "1"]:
                    _, s, t = fid2st(i + j + k)
                    fid_refresh[i + j + k] = fid_pre[str2idx(s)] * fid_post[str2idx(t)]
        
        self._err_pre = err_pre
        self._err_post = err_post
        self.fid_refresh = fid_refresh
        return
    

    def _random_spam(self, noise_scale):
        """Noises for spam error. Since state preparation error does not matter in our theory, we set SPAM error the same for simplicity.
        Terminating measurement errors is a special kind of Pauli channel.
        """
        rate_spam = {}
        for i in [["I"], ["X", "Y", "Z"]]:
            for j in [["I"], ["X", "Y", "Z"]]:
                rate = (1 + np.random.rand()) * noise_scale ** ((len(i) + len(j)) / 2 - 1)
                for ii in i:
                    for jj in j:
                        rate_spam[ii + jj] = rate
        rate_spam["II"] += 1 - np.sum([v for v in rate_spam.values()])
        err_spam = pauli_error(list(rate_spam.items()))

        self._err_spam = err_spam
        self.rate_spam = rate_spam
        return