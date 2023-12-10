from qiskit import QuantumCircuit, execute
import numpy as np

from noise import Noise
from utils import fid2st


class Experiment:
    def __init__(self, noise_scale):
        self.sim = Noise(noise_scale)


    def learn_single_fid(self, fid, shots=1000000, m_p=False):
        """Learn e+\partial e"""
        sgn, s, t = fid2st(fid)
        s_avg = 0
        for _ in range(100):
            circ = QuantumCircuit(3, 3)
            self.sim.noisy_sp(circ, s)
            if m_p:
                self.sim.noisy_mcm(circ, 0)
            else:
                self.sim.compiled_mcm(circ, 0)
            self.sim.noisy_m(circ, t, 1)
            counts = execute(circ, self.sim.backend_noisy, shots=int(shots / 100)).result().get_counts()
            for res in counts:
                count = counts[res]
                res_par = np.array([int(c) for c in res])[::-1]
                for i in range(2):
                    if t[i] == 'I':
                        res_par[2 - i] = 0
                s_avg += sgn * count * (-1) ** (res_par[0] * (int(fid[1]) + int(fid[2])) + res_par[1] + res_par[2])
        s_avg /= (int(shots / 100) * 100)
        
        circ2 = QuantumCircuit(2, 2)
        self.sim.noisy_sp(circ2, s)
        self.sim.noisy_m(circ2, s, 0)
        counts = execute(circ2, self.sim.backend_noisy, shots=shots).result().get_counts()
        t_avg = 0
        for res in counts:
            count = counts[res]
            res_par = np.array([int(c) for c in res])[::-1]
            for i in range(2):
                if s[i] == 'I':
                    res_par[1 - i] = 0
            t_avg += count * (-1) ** (res_par[0] + res_par[1])
        t_avg /= shots
        return s_avg, t_avg
    

    def _noiseless_concat(self, circ, t, s):
        """Append the single qubit Clifford gates."""
        for i in range(2):
            if t[i] != s[i]:
                if t[i] == 'I' or s[i] == 'I':
                    raise ValueError('Mismatched Pauli pattern.')
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
    

    def learn_multiple_fid(self, fids, shots=1000000, m_p=False): # add logic for m_p
        """Learn a path"""
        if m_p:
            mcm = self.sim.noisy_mcm
        else:
            mcm = self.sim.compiled_mcm
        l = len(fids)
        ss = [""] * l
        ts = [""] * l
        tot_sgn = 1
        for i in range(l):
            sgn, ss[i], ts[i] = fid2st(fids[i])
            tot_sgn *= sgn
        s_avg = 0
        for _ in range(100):
            circ = QuantumCircuit(3, l + 2)
            self.sim.noisy_sp(circ, ss[0])
            for i in range(l - 1):
                mcm(circ, i)
                self._noiseless_concat(circ, ts[i], ss[i + 1])
            mcm(circ, l - 1)
            self.sim.noisy_m(circ, ts[l - 1], l)
            counts = execute(circ, self.sim.backend_noisy, shots=int(shots / 100)).result().get_counts()
            for res in counts:
                count = counts[res]
                res_par = np.array([int(c) for c in res])[::-1]
                for i in range(2):
                    if ts[l - 1][i] == 'I':
                        res_par[l + 1 - i] = 0
                mxy = 0
                for i in range(l):
                    mxy += res_par[i] * (int(fids[i][1]) + int(fids[i][2]))
                s_avg += tot_sgn * count * (-1) ** (mxy + res_par[l] + res_par[l + 1])
        s_avg /= (int(shots / 100) * 100)
        
        circ2 = QuantumCircuit(2, 2)
        self.sim.noisy_sp(circ2, ss[0])
        self.sim.noisy_m(circ2, ss[0], 0)
        counts = execute(circ2, self.sim.backend_noisy, shots=shots).result().get_counts()
        t_avg = 0
        for res in counts:
            count = counts[res]
            res_par = np.array([int(c) for c in res])[::-1]
            for i in range(2):
                if ss[0][i] == 'I':
                    res_par[1 - i] = 0
            t_avg += count * (-1) ** (res_par[0] + res_par[1])
        t_avg /= shots
        return s_avg, t_avg