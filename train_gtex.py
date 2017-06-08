from sideinfo_release import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import sys
import argparse


then = timeit.default_timer()

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default = '',  help='data path')
parser.add_argument('--dim', type=int, default = 1,  help='dimension of data')
parser.add_argument('--init', type=int, default = 5,  help='number of inits')
parser.add_argument('--out', type=str, default = 'test',  help='output_directory')
parser.add_argument('--prefix', type=str, default = 'http://localhost:8888/files',  help='url prefix')
parser.add_argument('--alpha', type=float, default = 0.05,  help='fdr')
parser.add_argument('--fdr_scale', type=float, default = 1,  help='fd scale')
parser.add_argument('--mirror', type=float, default = 1,  help='mirror')
parser.add_argument('--net_scale', type=float, default = 1,  help='mirror')
parser.add_argument('--usedim', type=int, default = -1,  help='dimension of data')




opt = parser.parse_args()
print (opt)

fn = opt.data
dim = opt.dim




data = np.loadtxt(open(fn, "rb"), delimiter=",", skiprows=1)
x = data[:,0:dim]
p = data[:,dim]
h = data[:,dim+1]
n_samples = len(x)


if opt.usedim >= 0:
    x = x[:, opt.usedim]
    dim = 1
    
grids = None
x_prob = None

if dim == 1:
    max_x = np.max(x)
    min_x = np.min(x)
    x_prob = np.arange(min_x, max_x, (max_x - min_x)/1000.0)
    x_prob = x_prob.reshape((len(x_prob), 1))
    x_prob = Variable(torch.from_numpy(x_prob.astype(np.float32)))

elif dim == 2:
    max_x0 = np.max(x[:,0])
    min_x0 = np.min(x[:,0])
    max_x1 = np.max(x[:,1])
    min_x1 = np.min(x[:,1])
    x_prob0 = np.arange(min_x0, max_x0, (max_x0 - min_x0)/100.0)
    x_prob1 = np.arange(min_x1, max_x1, (max_x1 - min_x1)/100.0)
    X_grid, Y_grid = np.meshgrid(x_prob0, x_prob1)
    x_prob = Variable(torch.from_numpy(
    np.concatenate([[X_grid.flatten()], [Y_grid.flatten()]]).T.astype(np.float32)))
    grids = (X_grid, Y_grid)


if x_prob:
    print(x_prob.size())
    x_prob = x_prob.cuda()

#network = get_network(cuda = True, dim = dim)
#optimizer = optim.Adagrad(network.parameters(), lr = 0.01)


indices = np.random.permutation(x.shape[0])
A = [indices[:x.shape[0]/3], indices[x.shape[0]/3 : x.shape[0]/3*2], indices[x.shape[0]/3 * 2:]]
train = A
val = [A[1], A[2], A[0]]
test = [A[2], A[0], A[1]]
outputs = []
preds = []
gts = []

info = {}
info['filename'] = fn.replace('_', '\_')

loss_hists1 = []
loss_hists2 = []

efdr = np.zeros((3,3))
scales = np.zeros(3)

ninit = opt.init

bhp = BH(p, alpha = opt.alpha)[1]
lambda_param = 4/bhp
lambda_param = 5e4
print('lambda ', lambda_param)

if dim == 1:
    x = x.reshape((x.shape[0], 1))

    
def get_network_new(num_layers = 10, node_size = 10, dim = 1, scale = 1, cuda = False):
    
    
    class Model(nn.Module):
        def __init__(self, num_layers, node_size, dim, scale):
            super(Model, self).__init__()
            l = []
            l.append(nn.Linear(dim,node_size))
            l.append(nn.LeakyReLU(0.1))
            for i in range(num_layers - 2):
                l.append(nn.Linear(node_size,node_size))
                l.append(nn.LeakyReLU(0.1))

            l.append(nn.Linear(node_size,1))
            
            self.scale = scale
            self.layers = nn.Sequential(*l)


        def forward(self, x):
            x = self.layers(x)
            x = torch.mul(torch.exp(x) , self.scale)
            
            return x


    network = Model(num_layers, node_size, dim, scale)
    if cuda:
        return network.cuda()
    else:
        return network    
    
def init(x, p):

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=20)
    group = km.fit_predict(x)
    
    alpha = 0.1
    ths = []
    discs = 0
    for i in range(20):
        p_null = p[np.logical_and(group == i, p > 1-0.005)]
        p_alt = p[np.logical_and(group == i, p < 0.005)]
        th_high = 0.005
        th_low = 0

        for j in range(200):
            th = (th_high + th_low)/2
            #print th
            fd = np.sum(p_null > 1-th)
            td = np.sum(p_alt < th)
            #print fd, td*alpha, th
            if fd > td * alpha:
                th_high = th
            else:
                th_low = th

        ths.append(th)
        discs += td
        print td, fd, th
    ths = np.array(ths)
    
    dist = (x.repeat(20, axis = 1) - km.cluster_centers_.T ) ** 2
    s = np.sum(dist, axis = 1)
    prob = np.exp(-dist * 5) / np.expand_dims(np.sum(np.exp(-dist * 5), axis = 1),1)
    p_target = prob.dot(ths)
    
    return p_target

def init2(x, p):

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=20)
    group = km.fit_predict(x)
    
    alpha = 0.1
    ths = []
    discs = 0
    for i in range(20):
        p_null = p[np.logical_and(group == i, p > 1-0.005)]
        p_alt = p[np.logical_and(group == i, p < 0.005)]
        th_high = 0.005
        th_low = 0

        for j in range(200):
            th = (th_high + th_low)/2
            #print th
            fd = np.sum(p_null > 1-th)
            td = np.sum(p_alt < th)
            #print fd, td*alpha, th
            if fd > td * alpha:
                th_high = th
            else:
                th_low = th

        ths.append(th)
        discs += td
        print td, fd, th
    ths = np.array(ths)
    
    print ths
    print discs
    
    dist = (((np.expand_dims(x,2)).repeat(20, axis = 2) - km.cluster_centers_.T ) ** 2).sum(axis = 1)
    print dist.shape
    s = np.sum(dist, axis = 1)
    prob = np.exp(-dist * 5) / np.expand_dims(np.sum(np.exp(-dist * 5), axis = 1),1)
    p_target = prob.dot(ths)
    
    return p_target


for i in range(3):
    networks = []
    scores = []
    loss_hist1_array = []
    loss_hist2_array = []
    for j in range(ninit):
        
        train_idx = train[i]
        val_idx = val[i]
        test_idx = test[i]

        #network init
        #try:
        #    p_target = opt_threshold_multi(x[train_idx,:], p[train_idx], 10, alpha = opt.alpha)
        #except:
        print(BH(p[train_idx], alpha = opt.alpha, n = 10623893/3))
        bh_scale = BH(p[train_idx], alpha = opt.alpha, n = 10623893/3)[1]
        print(bh_scale)
        if dim == 1:
            p_target = init(x[train_idx,:], p[train_idx])
        elif dim >= 2:
            p_target = init2(x[train_idx,:], p[train_idx])
        network = get_network_new(num_layers = 10, cuda = True, dim = dim, scale = bh_scale)
        optimizer = optim.Adagrad(network.parameters(), lr = 0.01)
        
        #plt.figure()
        #plt.scatter(x, p_target)
        loss_hist = train_network_to_target_p(network, optimizer, x[train_idx,:], p_target, num_it = 2000, cuda= True, dim = dim)
        loss_hist2, s, s2 = train_network(network, optimizer, x[train_idx,:], p[train_idx], num_it = 8000, cuda = True, dim = dim, alpha = opt.alpha, lambda2_ = lambda_param, fdr_scale = opt.fdr_scale, mirror = opt.mirror, lambda_ = 10)

        loss_hist_np = np.array(loss_hist2)
        score = np.mean(loss_hist_np[-100:])
        print(j,score)
        networks.append(network)
        scores.append(score)
        loss_hist1_array.append(loss_hist)
        loss_hist2_array.append(loss_hist2)

    idx = np.argmin(np.array(scores))
    print idx

    loss_hist = loss_hist1_array[idx]
    loss_hist2 = loss_hist2_array[idx]
    network = networks[idx]

    loss_hists1.append(loss_hist)
    loss_hists2.append(loss_hist2)

    scale, efdr[i,1] = get_scale(network, x[val_idx,:], p[val_idx], cuda = True, lambda2_ = 5e12, fit = True, dim = dim, alpha = opt.alpha, fdr_scale = opt.fdr_scale, mirror = opt.mirror)
    _, efdr[i,2] = get_scale(network, x[test_idx,:], p[test_idx], cuda = True, lambda2_ = 5e12, scale = scale, dim = dim, alpha = opt.alpha, fdr_scale = opt.fdr_scale, mirror = opt.mirror)
    _, efdr[i,0] = get_scale(network, x[train_idx,:], p[train_idx], cuda = True, lambda2_ = 5e12, scale = scale, dim = dim, alpha = opt.alpha, fdr_scale = opt.fdr_scale, mirror = opt.mirror)

    scales[i] = scale

    n_samples = len(x[test_idx])
    x_input = Variable(torch.from_numpy(x[test_idx,:].astype(np.float32).reshape(n_samples ,dim))).cuda()
    p_input = Variable(torch.from_numpy(p[test_idx].astype(np.float32).reshape(n_samples ,1))).cuda()
    output = network.forward(x_input) * scale
    pred = (p_input < output).cpu().data.numpy()
    pred = pred[:,0].astype(np.float32)
    preds.append(pred)

    if not x_prob is None:
        outputs.append(network.forward(x_prob) * scale)

    gts.append(h[test_idx])


preds = np.concatenate(preds)
gts = np.concatenate(gts)

print sum(gts)
print sum(preds)
print sum(preds * gts)
print 1 - sum(preds * gts)/sum(preds)

info['number of ground truth discoveries'] = sum(gts)
info['number of discoveries'] = sum(preds)
info['set FDR'] = opt.alpha
info['actual FDR'] = 1 - sum(preds * gts)/sum(preds)
info['BH result'] = BH(p, alpha = opt.alpha, n = 10623893)
info['Storey BH result'] = Storey_BH(p, alpha = opt.alpha)
info['elapsed time'] = timeit.default_timer() - then


if x_prob:
    x_prob_data = x_prob.cpu().data.numpy()
    output_data =  [item.cpu().data.numpy() for item in outputs]
else:
    x_prob_data = None
    output_data = []

url = generate_report(x = x, p = p, h = h, out_dir = opt.out, url_prefix = opt.prefix, info = info, loss1 = loss_hists1, loss2 = loss_hists2, scales = scales, efdr = efdr, x_prob = x_prob_data, outputs = output_data, grids = grids)

print(url)
