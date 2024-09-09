"""Generate noisy circuit components that follows our noise model."""
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit.quantum_info import Operator, PTM, Pauli, random_quantum_channel
import numpy as np
from collections import defaultdict

from utils import str2idx, fid2st

class Noise:
    """Noisy circuit components that follows our noise model."""
    def __init__(self, noise_scale):
        """The constructor.
        @param self The object pointer.
        @param noise_scale The noise scale. The absolute value does not have a physical meaning.
        """
        # Errors will be appended after these two auxilary identity gates.
        self._id_gate = Operator(np.eye(4))
        self._id_gate3 = Operator(np.eye(8))
        # Generate noisy circuit components.
        self._random_qi(noise_scale)
        self._random_refresh(noise_scale) # To make random qi and random measure-and-prepare have similar error rates.
        self._random_spam(noise_scale * 0.25) # Terminating measurement error rates should be smaller than that of MCM's.
        # Label the noises so that we can attach them to gates.
        self.noise_model = NoiseModel()
        self.noise_model.add_basis_gates(['unitary'])
        self.noise_model.add_all_qubit_quantum_error(self._err_pre, 'pre')
        self.noise_model.add_all_qubit_quantum_error(self._err_post, 'post')
        self.noise_model.add_all_qubit_quantum_error(self._err_spam, 'spam')
        self.noise_model.add_all_qubit_quantum_error(self._err_qi, 'qi')
        self.backend_noisy = AerSimulator(noise_model=self.noise_model)
    

    def noisy_mcm(self, circ, cbit):
        """Append a noisy measure and prepare instrument to the circuit.
        We apply a Pauli noise, then an ideal MCM, followed by another Pauli noise.
        @param self The object pointer.
        @param circ The circuit to be appended.
        @param cbit The classical bit on which the measurement outcome should be stored.
        """
        circ.cnot(0, 1)
        circ.unitary(self._id_gate, [0, 1], label='pre')
        circ.measure(1, cbit)
        circ.unitary(self._id_gate, [0, 1], label='post')
    

    def compiled_mcm(self, circ, cbit):
        """Append a noisy compiled MCM to the circuit.
        The noise is a general quantum instrument and we perform randomized compiling on it.
        @param self The object pointer.
        @param circ The circuit to be appended.
        @param cbit The classical bit on which the measurement outcome should be stored.
        """
        # The circuit is described in Appendix A 1.
        # The alpha, beta, gamma and p here are the ones in Appendix A 1.
        p = np.random.choice(['I', 'X', 'Y', 'Z'])
        alpha, beta, gamma = np.random.randint(2, size=3)
        # The P otimes Z^beta X^alpha. Note that qiskit uses a reverse ordering of the qubits.
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
        # The G^dagger(P otimes Z^beta X^alpha). Note that qiskit uses a reverse ordering of the qubits. There might be a negative sign after conjugation.
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
        # To apply the helper channel, we need to introduce the third qubit which is initialized in 0.
        circ.reset(2)
        circ.cnot(0, 1)
        # The general quantum instrument is appended after the identity gate.
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
        """Append a noisy state preparation to the circuit.
        Though almost any initial state works for our protocol, it would be better for the initial state to have big overlap with the ideal operator.
        Here we assume that we prepare the eigenstate of the ideal operator and then apply a special Pauli noise (same as the terminating measurement nosie).
        @param self The object pointer.
        @param circ The circuit to be appended. It should be initalized to 0.
        @param p The Pauli operator that we want the initial state to have a large component of.
        """
        p = p[::-1] # Note that qiskit uses a reverse ordering of the qubits.
        for i in range(2):
            # For identity, an arbitrary stabilizer state is its +1 eigenstate.
            if p[i] == 'I':
                p = p[:i] + np.random.choice(['X', 'Y', 'Z']) + p[i + 1:]
            # Twist 0 to the corresponding eigenstates.
            if p[i] == 'X':
                circ.h(i)
            elif p[i] == 'Y':
                circ.h(i)
                circ.s(i)
        circ.unitary(self._id_gate, [0, 1], label='spam') # Append the state preparation noise.
    

    def noisy_m(self, circ, p, cbit):
        """Append a noisy terminating measurement to the circuit.
        We apply a special Pauli noise and then make an ideal terminating measurement.
        @param self The object pointer.
        @param circ The circuit to be appended.
        @param p The Pauli operator that we want to measure.
        @param cbit The classical bits on which the measurement outcome should be stored. Result will be stored at cbit and cbit+1.
        """
        p = p[::-1]
        circ.unitary(self._id_gate, [0, 1], label='spam') # Apply the terminating measurement noise.
        # If the operator we want to measure is I, then we will still make a computational basis measurement but will ignore the output.
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
        We use a helper channel to simulate a quantum instrument.
        The helper channel is Q_0 otimes I+Q_1 otimes X.
        If we initialize the third qubit in 0, apply the helper channel and then apply a perfect computational basis measurement on the third qubit and then trace it out, then the process is equivalent to applying the quantum instrument Q_0,Q_1.
        @param self The object pointer.
        @param noise_scale The noise scale.
        """
        # It is hard to satisfy complete positivity if we construct arbitrarily.
        # So we have to rely on some physical structures.
        # We compose noise before measurement + misreport error + noise after measurement together, and then average several instances to form a joint probability distribution.
        tot_q0 = np.zeros([16, 16], dtype=complex)
        tot_q1 = np.zeros([16, 16], dtype=complex)
        tot_weight = 0.0
        for _ in range(2):
            # misreport errors
            p01 = np.random.rand() * noise_scale
            p10 = np.random.rand() * noise_scale
            # The Pauli transfer matricies of the misreport error channel.
            q0 = np.kron(np.array([[1-p01+p10,0,0,1-p01-p10],[0,0,0,0],[0,0,0,0],[1-p01-p10,0,0,1-p01+p10]])/2, np.eye(4))
            q1 = np.kron(np.array([[1-p10+p01,0,0,-1+p10+p01],[0,0,0,0],[0,0,0,0],[-1+p10+p01,0,0,1-p10+p01]])/2, np.eye(4))
            # Noises before and after the measurement.
            n00 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            n01 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            n10 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            n11 = PTM(random_quantum_channel(input_dims=2, output_dims=2)).data * noise_scale + np.eye(4) * (1 - noise_scale)
            # Average the quantum instruments by averaging the Pauli transfer matricies.
            weight = np.random.rand()
            tot_weight += weight
            tot_q0 += np.kron(n00, n01) @ q0 @ np.kron(n10, n11) * weight
            tot_q1 += np.kron(n00, n01) @ q1 @ np.kron(n10, n11) * weight
        # To make the error more biased (i.e. to avoid having the same true values for many different Pauli fidelities), we mix the above instruments with some other instruments.
        # For this part we just set misreport error to 0.
        q0 = np.kron(np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2, np.eye(4))
        q1 = np.kron(np.array([[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]])/2, np.eye(4))
        # The new instrument is constructed by defining the Pauli error rates.
        e = ['X', 'Y']
        for i in range(2):
            # The noise (Pauli channel) after measurement.
            rate0 = {}
            rate0['X' + e[i]] = noise_scale * (4 + np.random.rand())
            rate0['II'] = 1 - rate0['X' + e[i]]
            n0 = PTM(pauli_error(list(rate0.items()))).data
            # The noise (Pauli channel) before measurement.
            rate1 = {}
            rate1['XI'] = noise_scale * 4 * i
            rate1['II'] = 1 - rate1['XI']
            n1 = PTM(pauli_error(list(rate1.items()))).data
            # Average the quantum instruments by averaging the Pauli transfer matricies.
            weight = np.random.rand() + 1
            tot_weight += weight
            tot_q0 += n0 @ q0 @ n1 * weight
            tot_q1 += n0 @ q1 @ n1 * weight
        tot_q0 /= tot_weight
        tot_q1 /= tot_weight
        # We conbine the Q_0 and Q_1 to get the helper channel.
        helper_chan = PTM(np.kron(np.eye(4), tot_q0) + np.kron(np.diag([1,1,-1,-1]), tot_q1))
        assert helper_chan.is_cptp()
        # Now we have the quantum instrument, calculate the true error rates and fidelities from the Pauli transfer matricies.
        rate = {}
        fid = defaultdict(float)
        channels = [tot_q0, tot_q1]
        # We use the extremely complicated Equation 45 from the paper Randomized compiling for subsystem measurements to do the conversion.
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

        # Store the helper channel and its true values that we finally get.
        self._err_qi = helper_chan
        self.fid_qi = fid
        self.rate_qi = rate
        return
    

    def _random_refresh(self, noise_scale):
        """Generate noises for measure-and-prepare MCM. Noises are Pauli channels.
        @param self The object pointer.
        @param noise_scale The noise scale.
        """
        # Generate the noise channel before measurement from Pauli error rates. In general errors with large support should be less likely to happen.
        rate_pre = {}
        for i in ["I", "X", "Y", "Z"]:
            for j in ["I", "X", "Y", "Z"]:
                rate_pre[i+j] = np.random.rand() * noise_scale ** ((i != 'I') + (j != 'I'))
        rate_pre["II"] += 1 - np.sum([v for v in rate_pre.values()])
        err_pre = pauli_error(list(rate_pre.items()))
        fid_pre = np.real_if_close(np.diag(PTM(err_pre).data))

        # Generate the noise channel before measurement from Pauli error rates. In general errors with large support should be less likely to happen.
        rate_post = {}
        for i in ["I", "X", "Y", "Z"]:
            for j in ["I", "X", "Y", "Z"]:
                rate_post[i+j] = np.random.rand() * noise_scale ** ((i != 'I') + (j != 'I'))
        rate_post["II"] += 1 - np.sum([v for v in rate_post.values()])
        err_post = pauli_error(list(rate_post.items()))
        fid_post = np.real_if_close(np.diag(PTM(err_post).data))

        # Calculate the true values of the Pauli fidelities of the measure and prepare channel.
        fid_refresh = {}
        for i in ["I", "X", "Y", "Z"]:
            for j in ["0", "1"]:
                for k in ["0", "1"]:
                    _, s, t = fid2st(i + j + k)
                    fid_refresh[i + j + k] = fid_pre[str2idx(s)] * fid_post[str2idx(t)]
        
        # Store the noise channels and the true Pauli fidelities.
        self._err_pre = err_pre
        self._err_post = err_post
        self.fid_refresh = fid_refresh
        return
    

    def _random_spam(self, noise_scale):
        """Generate noises for spam error. Since state preparation error does not matter in our theory, we set SPAM error the same for simplicity.
        Terminating measurement errors is a special kind of Pauli channel, as discussed in Appendix A 2.
        @param self The object pointer.
        @param noise_scale The noise scale.
        """
        # We generate the channel from Pauli error rates. Errors of the same support need to appear equally likely.
        rate_spam = {}
        for i in [["I"], ["X", "Y", "Z"]]:
            for j in [["I"], ["X", "Y", "Z"]]:
                rate = (1 + np.random.rand()) * noise_scale ** ((len(i) + len(j)) / 2 - 1)
                for ii in i:
                    for jj in j:
                        rate_spam[ii + jj] = rate
        rate_spam["II"] += 1 - np.sum([v for v in rate_spam.values()])
        err_spam = pauli_error(list(rate_spam.items()))

        # Store the noise channel and the true Pauli fidelities.
        self._err_spam = err_spam
        self.rate_spam = rate_spam
        return