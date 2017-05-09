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

network = get_network(cuda = True)
optimizer = optim.Adagrad(network.parameters(), lr = 0.01)


indices = np.random.permutation(x.shape[0])
A = [indices[:x.shape[0]/3], indices[x.shape[0]/3 : x.shape[0]/3*2], indices[x.shape[0]/3 * 2:]]
train = A
val = [A[1], A[2], A[0]]
test = [A[2], A[0], A[1]]
outputs = []
preds = []
gts = []
for i in range(3):
    network = get_network(num_layers = 10).cuda()
    optimizer = optim.Adagrad(network.parameters(), lr = 0.01)
    train_idx = train[i]
    val_idx = val[i]
    test_idx = test[i]
    
    #network init
    p_target = opt_threshold(x[train_idx], p[train_idx], 10)
    #plt.figure()
    #plt.scatter(x, p_target)
    loss_hist = train_network_to_target_p(network, optimizer, x[train_idx], p_target, num_it = 6000, cuda= True)
    loss_hist2, s, s2 = train_network(network, optimizer, x[train_idx], p[train_idx], num_it = 9000, cuda = True)
    
    scale = get_scale(network, x[val_idx], p[val_idx], cuda = True, lambda2_ = 5e3, fit = True)
    _ = get_scale(network, x[test_idx], p[test_idx], cuda = True, lambda2_ = 5e3, scale = scale)

    n_samples = len(x[test_idx])
    x_input = Variable(torch.from_numpy(x[test_idx].astype(np.float32).reshape(n_samples ,1))).cuda()
    p_input = Variable(torch.from_numpy(p[test_idx].astype(np.float32).reshape(n_samples ,1))).cuda()
    output = network.forward(x_input) * scale
    pred = (p_input < output).cpu().data.numpy()
    pred = pred[:,0].astype(np.float32)
    preds.append(pred)
    
    x2 = np.arange(0, 5, 0.01)
    n_samples = len(x2)
    x_input = Variable(torch.from_numpy(x2.astype(np.float32).reshape(n_samples ,1))).cuda()
    outputs.append(network.forward(x_input) * scale)
    gts.append(h[test_idx])
    
    
preds = np.concatenate(preds)
gts = np.concatenate(gts)
print sum(gts)
print sum(preds)
print sum(preds * gts)
print 1 - sum(preds * gts)/sum(preds)