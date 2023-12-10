import numpy as np
import matplotlib.pyplot as plt

from experiment import Experiment

#np.random.seed(114514) # Do NOT set random seeds!
experiment = Experiment(0.003)
shots = 1000000
bootstrap = 100
reGen = True

# Learning the cycle basis.
cycle_basis = [["Z00"], ["X01"], ["X11"], ["Y01"], ["Y11"],
               ["I11", "Z11"], ["Z01", "X00"], ["Z01", "X10"], ["Z01", "Y00"], ["Z01", "Y10"],
               ["Z10", "Z01", "I11"], ["I01", "Z10", "Z01", "I10"]] # \lambda^I_{00} excluded
if reGen:
    learned_mean = []
    boosted_std = []
    true_mean = []
    for targ in cycle_basis:
        print(targ)
        repeat = int(12 / len(targ))
        s, t = experiment.learn_multiple_fid(targ * repeat, shots=shots)
        ss = (np.random.binomial(shots, (s + 1) / 2, bootstrap) / shots) * 2 - 1
        tt = (np.random.binomial(shots, (t + 1) / 2, bootstrap) / shots) * 2 - 1
        learned_mean.append((s / t) ** (1. / 12))
        boosted_std.append(np.std((ss / tt) ** (1. / 12)))
        true_fid = 1
        for fid in targ:
            true_fid *= experiment.sim.fid_qi[fid]
        true_mean.append(true_fid ** (1. / len(targ)))
    np.savetxt('cycle.txt', (learned_mean, boosted_std, true_mean))
else:
    learned_mean, boosted_std, true_mean = np.loadtxt('cycle.txt')


def fmt_label(cycle):
    ans = r"$\{"
    for fid in cycle:
        ans += r"\lambda^" + fid[0] + r"_{" + fid[1:] + r"},"
    ans = ans[:-1] + r"\}$"
    return ans


plt.rcParams['text.usetex'] = True
plt.figure(figsize=[10, 4.8])
plt.errorbar([fmt_label(cycle) for cycle in cycle_basis], learned_mean, boosted_std, fmt='o', markersize=4, capsize=5, label='Learned Values', zorder=1)
plt.scatter([fmt_label(cycle) for cycle in cycle_basis], true_mean, marker='^', s=16, c='r', label='True Values', zorder=2)
plt.grid(True)
plt.xticks(rotation=-45, fontsize=10, va='top', ha='left')
plt.legend()
plt.title('Average Fidelity')
plt.savefig('cycle.pdf', bbox_inches='tight')

# Testing the independence of measurement and state preparation.
# First check that the two settings has similar error rates.
print(np.sum(list(experiment.sim.fid_qi.values())) / 16, np.sum(list(experiment.sim.fid_refresh.values())) / 16)


def indepTest(m_p):
    f_mean = []
    f_std = []
    f_true = []
    for p in ['I', 'X', 'Y', 'Z']:
        rpos = []
        rneg = []
        for subscript in ["00", "11"]:
            s, t = experiment.learn_single_fid(p + subscript, shots=shots, m_p=m_p)
            rpos.append(s)
            rneg.append(t)
        for subscript in ["01", "10"]:
            s, t = experiment.learn_single_fid(p + subscript, shots=shots, m_p=m_p)
            rpos.append(t)
            rneg.append(s)
        f_mean.append(np.sum(np.log(rpos)) - np.sum(np.log(rneg)))
        rposs = []
        rnegs = []
        for s in rpos:
            rposs.append((np.random.binomial(shots, (s + 1) / 2, bootstrap) / shots) * 2 - 1)
        for t in rneg:
            rnegs.append((np.random.binomial(shots, (t + 1) / 2, bootstrap) / shots) * 2 - 1)
        f_std.append(np.std(np.sum(np.log(rposs), axis=0) - np.sum(np.log(rnegs), axis=0)))
        if m_p:
            fid = experiment.sim.fid_refresh
        else:
            fid = experiment.sim.fid_qi
        f_true.append((np.log(fid[p + "00"]) + np.log(fid[p + "11"])
                    - np.log(fid[p + "01"]) - np.log(fid[p + "10"])))
    return f_mean, f_std, f_true


if reGen:
    qi_mean, qi_std, qi_true = indepTest(False)
    refresh_mean, refresh_std, refresh_true = indepTest(True)
    np.savetxt('indep.txt', (qi_mean, qi_std, qi_true, refresh_mean, refresh_std, refresh_true))
else:
    qi_mean, qi_std, qi_true, refresh_mean, refresh_std, refresh_true = np.loadtxt('indep.txt')
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12.8, 4.8))
ax1.errorbar(['I', 'X', 'Y', 'Z'], qi_mean, qi_std, fmt='o', markersize=4, capsize=5, label='Learned Values', zorder=1)
ax1.scatter(['I', 'X', 'Y', 'Z'], qi_true, marker='^', s=16, c='r', label='True Values', zorder=2)
ax1.grid(True)
ax1.legend()
ax1.set_title("General Quantum Instrument")
ax2.errorbar(['I', 'X', 'Y', 'Z'], refresh_mean, refresh_std, fmt='o', markersize=4, capsize=5, label='Learned Values', zorder=1)
ax2.scatter(['I', 'X', 'Y', 'Z'], refresh_true, marker='^', s=16, c='r', label='True Values', zorder=2)
ax2.grid(True)
ax2.yaxis.set_tick_params(labelbottom=True)
ax2.legend()
ax2.set_title("Measure and Prepare")
plt.savefig('indep.pdf', bbox_inches='tight')

# Learning the error rate p^I_{11}.
print(experiment.sim.rate_qi["I11"])
rpos = []
rneg = []
s, t = experiment.learn_multiple_fid(['X11','Y11', 'Y00', 'Z01', 'X00', 'Z00', 'Z01', 'I11', 'Z11'])
rpos.append(s)
rneg.append(t)
s, t = experiment.learn_multiple_fid(['Y01','X10', 'Z01', 'X01', 'Y10', 'Z01', 'I10', 'I01', 'Z10', 'Z01'])
rpos.append(t)
rneg.append(s)
rposs = []
rnegs = []
for s in rpos:
    rposs.append((np.random.binomial(shots, (s + 1) / 2, bootstrap) / shots) * 2 - 1)
for t in rneg:
    rnegs.append((np.random.binomial(shots, (t + 1) / 2, bootstrap) / shots) * 2 - 1)
print((np.sum(np.log(rpos)) - np.sum(np.log(rneg))) / 16)
print(np.std(np.sum(np.log(rposs), axis=0) - np.sum(np.log(rnegs), axis=0)))