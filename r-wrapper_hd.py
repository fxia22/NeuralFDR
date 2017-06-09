
# coding: utf-8

# In[1]:

from numpy import *
import scipy as sp
from pandas import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

import sys

import numpy as np

dim = int(sys.argv[2])
fn = sys.argv[1]
data = np.loadtxt(open(fn, "rb"), delimiter=",", skiprows=1)
x = data[:,0:dim]
p = data[:,dim]
h = data[:,dim+1]
n_samples = len(x)

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 20)
group = km.fit_predict(x)
print(group.shape)

with open(fn+'.ihw', 'w') as f:
    f.write('group, p_value, h\n')
    for i in range(len(x)):
        f.write("{}, {}, {}\n".format(group[i],p[i], h[i]))


ro.r('data = read.csv(\'{}\', head=TRUE)'.format(fn + '.ihw'))


# In[3]:

ro.r('source("https://bioconductor.org/biocLite.R")')


# In[4]:

ro.r("biocLite(\"IHW\")")


# In[6]:

ro.r("library(\"IHW\")")


# In[7]:

ro.r("ihwRes <- ihw(p_value ~ group ,  data = data, alpha = 0.1)")


# In[9]:

res = ro.r("rejections(ihwRes)")


# In[13]:

print(np.array(res))
print(sys.argv[1])




