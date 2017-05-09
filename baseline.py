from sideinfo_release import *
import matplotlib.pyplot as plt
import numpy as np

import sys

fn = sys.argv[1]
if len(sys.argv) > 2:
    dim = int(sys.argv[2])
else:
    dim = 1


data = np.loadtxt(open(fn, "rb"), delimiter=",", skiprows=1)
x = data[:,0:dim]
p = data[:,dim]
h = data[:,dim+1]
n_samples = len(x)

print(BH(p))
print(Storey_BH(p))
