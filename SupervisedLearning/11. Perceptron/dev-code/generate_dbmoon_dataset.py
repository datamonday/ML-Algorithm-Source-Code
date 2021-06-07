# -*- coding: utf-8 -*-
import numpy as np


class moon_data_class(object):
  def __init__(self,N,d,r,w):
    self.N=N
    self.w=w
    
    self.d=d
    self.r=r
   
  def sgn(self,x):
      
    if(x>0):
      return 1;
    else:
      return -1;
     
  def sig(self,x):
    return 1.0/(1+np.exp(x))
   
     
  def dbmoon(self):
    N1 = 10*self.N
    N = self.N
    r = self.r
    w2 = self.w/2
    d = self.d
    done = True
    data = np.empty(0)
    
    while done:
      #generate Rectangular data
      tmp_x = 2*(r+w2)*(np.random.random([N1, 1])-0.5)
      tmp_y = (r+w2)*np.random.random([N1, 1])
      tmp = np.concatenate((tmp_x, tmp_y), axis=1)
      tmp_ds = np.sqrt(tmp_x*tmp_x + tmp_y*tmp_y)
      
      #generate double moon data ---upper
      idx = np.logical_and(tmp_ds > (r-w2), tmp_ds < (r+w2))
      idx = (idx.nonzero())[0]
    
      if data.shape[0] == 0:
        data = tmp.take(idx, axis=0)
      else:
        data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
      if data.shape[0] >= N:
        done = False
        
    #print (data)
    db_moon = data[0:N, :]
    #print (db_moon)
    
    #generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    
    return db_moon