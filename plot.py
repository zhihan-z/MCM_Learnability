"""Use our protocol to perform various tasks. Generate plots for the numerical simulations."""
import numpy as np
import matplotlib.pyplot as plt

from experiment import Experiment

experiment = Experiment(0.0030) # Set a noise scale.
bootstrap = 100 # Number of resamplings during bootstrapping.
reGen = True # Set to True to generate new data.

# Learning the cycle basis (Part a).
shots = 10000
cycle_basis = [["Z00"], ["X01"], ["X11"], ["Y01"], ["Y11"],
               ["I11", "Z11"], ["Z01", "X00"], ["Z01", "X10"], ["Z01", "Y00"], ["Z01", "Y10"],
               ["Z10", "Z01", "I11"], ["I01", "Z10", "Z01", "I10"]] # The cycle basis, lambda^I_{00} excluded
if reGen:
    learned_mean = []
    boosted_std = []
    true_mean = []
    # Learn the cycles one by one.
    for targ in cycle_basis:
        print(targ)
        repeat = int(12 / len(targ)) # Repeat the cycles to make them all of length 12.
        s, t = experiment.learn_multiple_fid(targ * repeat, shots=shots) # Run our protocol.
        # Bootstrappping. Note that s and t are averages of random +1 and -1s. We resample them and take average.
        ss = (np.random.binomial(shots, (s + 1) / 2, bootstrap) / shots) * 2 - 1
        tt = (np.random.binomial(shots, (t + 1) / 2, bootstrap) / shots) * 2 - 1
        learned_mean.append((s / t) ** (1. / 12)) # The learned value.
        boosted_std.append(np.std((ss / tt) ** (1. / 12))) # The std estimated from bootstrapping.
        # Calculate the true values.
        true_fid = 1
        for fid in targ:
            true_fid *= experiment.sim.fid_qi[fid]
        true_mean.append(true_fid ** (1. / len(targ)))
    np.savetxt('cycle.txt', (learned_mean, boosted_std, true_mean))
else:
    learned_mean, boosted_std, true_mean = np.loadtxt('cycle.txt')


# Make the plot (Figure 7).
def fmt_label(cycle):
    """Output the cycle in latex format for x-axis labels.
    @param cycle The cycle.
    """
    ans = r"$\{"
    for fid in cycle:
        ans += r"\lambda^" + fid[0] + r"_{" + fid[1:] + r"},"
    ans = ans[:-1] + r"\}$"
    return ans


plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = "xx-large"
plt.rcParams['axes.labelsize'] = "x-large"
plt.rcParams['legend.fontsize'] = "large"
plt.figure(figsize=[10, 4.8])
plt.errorbar([fmt_label(cycle) for cycle in cycle_basis], learned_mean, boosted_std, fmt='o', markersize=4, capsize=5, label='learned values', zorder=1)
plt.scatter([fmt_label(cycle) for cycle in cycle_basis], true_mean, marker='^', s=16, c='r', label='true values', zorder=2)
plt.grid(True)
plt.xticks(rotation=-45, fontsize=10, va='top', ha='left')
plt.legend()
plt.xlabel('cycle')
plt.ylabel('average fidelity')
plt.title('Learned geometric averages of fidelities in cycles')
plt.savefig('cycle.pdf', bbox_inches='tight')

# Testing the independence of measurement and state preparation (Part b).
# First check that the two settings has similar error rates.
print(np.sum(list(experiment.sim.fid_qi.values())) / 16, np.sum(list(experiment.sim.fid_refresh.values())) / 16)
shots = 2000000


def indepTest(m_p):
    """Estimate the correlations.
    @param m_p True for simulating the protocol on the measure and prepare channel, False for simulating the protocol on the compiled MCM.
    """
    f_mean = []
    f_std = []
    f_true = []
    # Estimate the correlations one by one.
    for p in ['I', 'X', 'Y', 'Z']:
        rpos = []
        rneg = []
        # Learn the 4 log fidelities separately.
        # These two log fidelities should be added.
        for subscript in ["00", "11"]:
            s, t = experiment.learn_single_fid(p + subscript, shots=shots, m_p=m_p)
            rpos.append(s)
            rneg.append(t)
        # These two log fidelities should be subtracted.
        for subscript in ["01", "10"]:
            s, t = experiment.learn_single_fid(p + subscript, shots=shots, m_p=m_p)
            rpos.append(t)
            rneg.append(s)
        f_mean.append(np.sum(np.log(rpos)) - np.sum(np.log(rneg))) # The learned value.
        # Bootstrapping.
        rposs = []
        rnegs = []
        for s in rpos:
            rposs.append((np.random.binomial(shots, (s + 1) / 2, bootstrap) / shots) * 2 - 1)
        for t in rneg:
            rnegs.append((np.random.binomial(shots, (t + 1) / 2, bootstrap) / shots) * 2 - 1)
        f_std.append(np.std(np.sum(np.log(rposs), axis=0) - np.sum(np.log(rnegs), axis=0)))
        # Calculate the true values.
        if m_p:
            fid = experiment.sim.fid_refresh
        else:
            fid = experiment.sim.fid_qi
        f_true.append((np.log(fid[p + "00"]) + np.log(fid[p + "11"])
                    - np.log(fid[p + "01"]) - np.log(fid[p + "10"])))
    return f_mean, f_std, f_true


# Make the plot (Figure 8).
if reGen:
    qi_mean, qi_std, qi_true = indepTest(False)
    refresh_mean, refresh_std, refresh_true = indepTest(True)
    np.savetxt('indep.txt', (qi_mean, qi_std, qi_true, refresh_mean, refresh_std, refresh_true))
else:
    qi_mean, qi_std, qi_true, refresh_mean, refresh_std, refresh_true = np.loadtxt('indep.txt')
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12.8, 4.8), constrained_layout=True)
ax1.errorbar(['I', 'X', 'Y', 'Z'], qi_mean, qi_std, fmt='o', markersize=4, capsize=5, label='learned values', zorder=1)
ax1.scatter(['I', 'X', 'Y', 'Z'], qi_true, marker='^', s=16, c='r', label='true values', zorder=2)
ax1.grid(True)
ax1.legend()
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_title("General compiled MCM")
ax1.set_xlabel(r'$Q$')
ax1.set_ylabel(r'$c^Q_{0,1,0,1}$')
ax2.errorbar(['I', 'X', 'Y', 'Z'], refresh_mean, refresh_std, fmt='o', markersize=4, capsize=5, label='learned values', zorder=1)
ax2.scatter(['I', 'X', 'Y', 'Z'], refresh_true, marker='^', s=16, c='r', label='true values', zorder=2)
ax2.grid(True)
ax2.yaxis.set_tick_params(labelbottom=True, labelsize=15)
ax2.legend()
ax2.set_title("Measure and prepare instrument")
ax2.set_xlabel(r'$Q$')
ax2.set_ylabel(r'$c^Q_{0,1,0,1}$')
plt.savefig('indep.pdf', bbox_inches='tight')

# Learning the error rate p^I_{11} (Part c).
if reGen:
    shots = 10000
    print(experiment.sim.rate_qi["I11"]) # The true value.
    apxval = 0
    for e in ['X11','Y11', 'Y00', 'Z01', 'X00', 'Z00', 'Z01', 'I11', 'Z11']:
        apxval += np.log(experiment.sim.fid_qi[e])
    for e in ['Y01','X10', 'Z01', 'X01', 'Y10', 'Z01', 'I10', 'I01', 'Z10', 'Z01']:
        apxval -= np.log(experiment.sim.fid_qi[e])
    print(apxval / 16) # 'True' value under first order approximation.
    ress = []
    for _ in range(10):
        # Learn the 2 cycles in Equation 46.
        res = 0
        s, t = experiment.learn_multiple_fid(['X11','Y11', 'Y00', 'Z01', 'X00', 'Z00', 'Z01', 'I11', 'Z11'])
        res += np.log(s / t)
        s, t = experiment.learn_multiple_fid(['Y01','X10', 'Z01', 'X01', 'Y10', 'Z01', 'I10', 'I01', 'Z10', 'Z01'])
        res -= np.log(s / t)
        res /= 16
        ress.append(res)
    print(np.average(ress)) # The learned value.
    print(np.std(ress) / np.sqrt(10 - 1)) # The estimated std.