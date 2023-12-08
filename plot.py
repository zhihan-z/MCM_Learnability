import numpy as np

from experiment import Experiment

np.random.seed(114514)
experiment = Experiment(0.002)

# Learn the cycle basis.
cycle_basis = [["I00"], ["Z00"], ["X01"], ["X11"], ["Y01"], ["Y11"],
               ["I11", "Z11"], ["Z01", "X00"], ["Z01", "X10"], ["Z01", "Y00"], ["Z01", "Y10"],
               ["Z10", "Z01", "I11"], ["I01", "Z10", "Z01", "I10"]]
for targ in cycle_basis:
    print(targ)
    print(np.exp(experiment.learn_multiple_fid(targ)))
    true_fid = 1
    for fid in targ:
        true_fid *= experiment.sim.fid_qi[fid]
    print(true_fid)

# Testing the independence of measurement and state preparation.
# First check that the two settings has similar error rates.
print(np.sum(list(experiment.sim.fid_qi.values())) / 16, np.sum(list(experiment.sim.fid_refresh.values())) / 16)

for p in ['I', 'X', 'Y', 'Z']:
    print(experiment.learn_single_fid(p + "00") + experiment.learn_single_fid(p + "11")
          - experiment.learn_single_fid(p + "01") - experiment.learn_single_fid(p + "10"))

for p in ['I', 'X', 'Y', 'Z']:
    print(experiment.learn_single_fid(p + "00", m_p=True) + experiment.learn_single_fid(p + "11", m_p=True)
          - experiment.learn_single_fid(p + "01", m_p=True) - experiment.learn_single_fid(p + "10", m_p=True))

print((experiment.learn_multiple_fid(['X11','Y11', 'Y00', 'Z01', 'X00', 'Z00', 'Z01', 'I11', 'Z11'])
       - experiment.learn_multiple_fid(['Y01','X10', 'Z01', 'X01', 'Y10', 'Z01', 'I10', 'I01', 'Z10', 'Z01'])) / 16)
print(experiment.sim.rate_qi["I11"])