
# coding: utf-8

# In[1]:

from numpy import *
import scipy as sp
from pandas import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

import sys

# In[2]:

ro.r('data = read.csv(\'{}\', head=TRUE)'.format(sys.argv[1]))


# In[3]:

ro.r('source("https://bioconductor.org/biocLite.R")')


# In[4]:

ro.r("biocLite(\"IHW\")")


# In[6]:

ro.r("library(\"IHW\")")


# In[7]:

ro.r("ihwRes <- ihw(p_value ~ x_value ,  data = data, alpha = 0.05)")


# In[9]:

res = ro.r("rejections(ihwRes)")


# In[13]:

print(np.array(res))


# In[ ]:



