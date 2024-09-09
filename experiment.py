"""Simulate our (modified) protocol. See Section IV C of our paper."""
from qiskit import QuantumCircuit, execute
import numpy as np

from noise import Noise
from utils import fid2st


class Experiment:
    """Our protocols."""
    def __init__(self, noise_scale):
        """The constructor.
        @param self The object pointer.
        @param noise_scale The noise scale. The absolute value does not have a physical meaning.
        """
        self.sim = Noise(noise_scale)


    def learn_single_fid(self, fid, shots=100000, m_p=False):
        """Learn e+partial e for an edge e.
        This is a special case of our protocol.
        There is only one edge, so there is only one MCM in the main circuit and there is no interleaving single qubit gates.
        @param self The object pointer.
        @param fid The edge e. e^Q_{x,y} should be input as the string Qxy.
        @param shots The number of shots used for both the main circuit and the auxilary circuit.
        @param m_p True for simulating the protocol on the measure and prepare channel, False for simulating the protocol on the compiled MCM.
        """
        sgn, s, t = fid2st(fid) # Get the Paulis we need for the initial state and the Pauli we should measure for terminating measurement.
        s_avg = 0
        # We use 100 compiled circuits for the main circuit.
        for _ in range(100):
            circ = QuantumCircuit(3, 3)
            self.sim.noisy_sp(circ, s) # Noisily prepare the +1 eigenstate of G^dagger(Q otimes Z^x)
            # Apply the MCM.
            if m_p:
                self.sim.noisy_mcm(circ, 0)
            else:
                self.sim.compiled_mcm(circ, 0)
            self.sim.noisy_m(circ, t, 1) # Noisy terminating measurement of Q otimes Z^y.
            # Now simulate the experiment.
            counts = execute(circ, self.sim.backend_noisy, shots=int(shots / 100)).result().get_counts()
            # We get 1 bit for the MCM result and 2 bits for the terminating measurement result.
            for res in counts:
                count = counts[res] # res is the result (reversed order in qiskit) and count is the number of times it is observed.
                res_par = np.array([int(c) for c in res])[::-1]
                # Ignore the bit in terminating measurement if what we actually want to measure is I. Note that t is in reversed order.
                for i in range(2):
                    if t[i] == 'I':
                        res_par[2 - i] = 0
                # The classical data processing (Step 4). res_par[0] is the m, fid[1] and fid[2] are x and y, respectively. res_par[1] and res_par[2] together gives the terminating measurement result r. Note that we need to take the sign into account.
                s_avg += sgn * count * (-1) ** (res_par[0] * (int(fid[1]) + int(fid[2])) + res_par[1] + res_par[2])
        s_avg /= (int(shots / 100) * 100) # This s_avg is the s in our protocol.
        
        # The auxilary circuit.
        circ2 = QuantumCircuit(2, 2)
        self.sim.noisy_sp(circ2, s) # Noisy state preparation.
        self.sim.noisy_m(circ2, s, 0) # Followed directly by the noisy terminating measurement.
        # Now simulate the experiment.
        counts = execute(circ2, self.sim.backend_noisy, shots=shots).result().get_counts()
        # We get 2 bits for the terminating measurement result.
        t_avg = 0
        for res in counts:
            count = counts[res] # res is the result (reversed order in qiskit) and count is the number of times it is observed.
            res_par = np.array([int(c) for c in res])[::-1]
            # Ignore the bit in terminating measurement if what we actually want to measure is I. Now we are measuring s and note that s is in reversed order.
            for i in range(2):
                if s[i] == 'I':
                    res_par[1 - i] = 0
            t_avg += count * (-1) ** (res_par[0] + res_par[1]) # Just calculate the expectation of the terminating measurement result.
        t_avg /= shots # This t_avg is the t in our protocol.
        return s_avg, t_avg # No logarithm is taken.
    

    def _noiseless_concat(self, circ, t, s):
        """Determine and append the single qubit Clifford gates to the circuit.
        @param self The object pointer.
        @param circ The circuit to be appended.
        @param t The ending Pauli string of the last edge (Q otimes Z^{y_i})
        @param s The starting Pauli string of the next edge (G^dagger(Q otimes Z^{x_{i+1}}))
        """
        # Determine case by case what the interleaving single qubit Clifford gates should be. Special care should be taken to avoid introducing signs.
        for i in range(2):
            if t[i] != s[i]:
                if t[i] == 'I' or s[i] == 'I':
                    raise ValueError('Mismatched Pauli pattern.') # The pattern of t and s should be the same.
                if t[i] == 'X':
                    if s[i] == 'Y':
                        circ.s(1 - i)
                    else:
                        circ.h(1 - i)
                elif t[i] == 'Y':
                    if s[i] == 'Z':
                        circ.sdg(1 - i)
                        circ.h(1 - i)
                    else:
                        circ.sdg(1 - i)
                else:
                    if s[i] == 'X':
                        circ.h(1 - i)
                    else:
                        circ.h(1 - i)
                        circ.s(1 - i)
        return
    

    def learn_multiple_fid(self, fids, shots=100000, m_p=False):
        """Learn a path together with its boundary.
        This is the generic version of our protocol.
        @param self The object pointer.
        @param fids The path. Presented as a list of strings for the edges along the path. Edge e^Q_{x,y} should appear as the string Qxy in the list.
        @param shots The number of shots used for both the main circuit and the auxilary circuit.
        @param m_p True for simulating the protocol on the measure and prepare channel, False for simulating the protocol on the compiled MCM.
        """
        # Choose the MCM depending on m_p.
        if m_p:
            mcm = self.sim.noisy_mcm
        else:
            mcm = self.sim.compiled_mcm
        l = len(fids) # The l in our protocol.
        ss = [""] * l
        ts = [""] * l
        tot_sgn = 1
        # Find the starting and ending Pauli strings for each edge. Track the overall sign at the same time.
        for i in range(l):
            sgn, ss[i], ts[i] = fid2st(fids[i])
            tot_sgn *= sgn
        s_avg = 0
        for _ in range(100):
            # Construct the circuit.
            circ = QuantumCircuit(3, l + 2)
            self.sim.noisy_sp(circ, ss[0]) # Noisily prepare the +1 eigenstate of G^dagger(Q_1 otimes Z^{x_1})
            for i in range(l - 1):
                mcm(circ, i) # Apply the MCM.
                self._noiseless_concat(circ, ts[i], ss[i + 1]) # Apply the interleaving single qubit Clifford gates.
            mcm(circ, l - 1) # Apply the last MCM.
            self.sim.noisy_m(circ, ts[l - 1], l) # Noisy terminating measurement of Q_l otimes Z^{y_l}.
            # Now simulate the experiment.
            counts = execute(circ, self.sim.backend_noisy, shots=int(shots / 100)).result().get_counts()
            # We get l bits for the MCM result and 2 bits for the terminating measurement result.
            for res in counts:
                count = counts[res] # res is the result (reversed order in qiskit) and count is the number of times it is observed.
                res_par = np.array([int(c) for c in res])[::-1]
                # Ignore the bit in terminating measurement if what we actually want to measure is I. Note that t is in reversed order.
                for i in range(2):
                    if ts[l - 1][i] == 'I':
                        res_par[l + 1 - i] = 0
                # The classical data processing (Step 4). The first l entrices of res_par are the m_i, fid[][1] and fid[][2] are x and y, respectively. res_par[l] and res_par[l+1] together gives the terminating measurement result r. Note that we need to take the sign into account.
                mxy = 0
                for i in range(l):
                    mxy += res_par[i] * (int(fids[i][1]) + int(fids[i][2]))
                s_avg += tot_sgn * count * (-1) ** (mxy + res_par[l] + res_par[l + 1])
        s_avg /= (int(shots / 100) * 100) # This s_avg is the s in our protocol.
        
        # The auxilary circuit.
        circ2 = QuantumCircuit(2, 2)
        self.sim.noisy_sp(circ2, ss[0]) # Noisy state preparation.
        self.sim.noisy_m(circ2, ss[0], 0) # Followed directly by the noisy terminating measurement.
        # Now simulate the experiment.
        counts = execute(circ2, self.sim.backend_noisy, shots=shots).result().get_counts()
        # We get 2 bits for the terminating measurement result.
        t_avg = 0
        for res in counts:
            count = counts[res] # res is the result (reversed order in qiskit) and count is the number of times it is observed.
            res_par = np.array([int(c) for c in res])[::-1]
            # Ignore the bit in terminating measurement if what we actually want to measure is I.
            for i in range(2):
                if ss[0][i] == 'I':
                    res_par[1 - i] = 0
            t_avg += count * (-1) ** (res_par[0] + res_par[1]) # Just calculate the expectation of the terminating measurement result.
        t_avg /= shots # This t_avg is the t in our protocol.
        return s_avg, t_avg # No logarithm is taken.