import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from doubleml import DoubleMLData,DoubleMLPLR
from sklearn.base import clone
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

np.random.seed(3141)

data_frame = pd.read_csv("crime_vio.csv")
data_frame = data_frame.drop(data_frame.columns[:3], axis=1)
data = data_frame.values

n, p = data.shape
XD = data[:,:-2]
y1 = data[:,-2]
y2 = data[:,-1]
y1 = y1/1000
y2 = y2/1000

ss = StandardScaler()
XD_s = ss.fit_transform(XD)

dim = p-2
coefs = np.zeros(dim)
ses = np.zeros(dim)

for i in range(dim):
	print(i)
	index_d = i
	d = XD_s[:,index_d]
	X = np.delete(XD_s, index_d, axis=1)

	dml_data = DoubleMLData.from_arrays(X, y2, d)
	learner_l = LassoCV(max_iter=20000)
	#learner_m = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', max_depth= 4)
	ml_l = clone(learner_l)
	ml_m = clone(learner_l)
	#ml_g = clone(learner)

	dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)
	dml_plr.fit()
	coefs[i] = dml_plr.coef
	ses[i] = dml_plr.se


sign_vars = []
for i in range(dim):
	if coefs[i]<=0 and coefs[i]+1.96*ses[i]<=0:
		sign_vars.append(i)
	if coefs[i]>0 and coefs[i]-1.96*ses[i]>0:
		sign_vars.append(i)

sign_vars = []
for i in range(dim):
	if coefs[i]<=0 and coefs[i]+1.96*ses[i]<=0:
		sign_vars.append(i)
	if coefs[i]>0 and coefs[i]-1.96*ses[i]>0:
		sign_vars.append(i)

indexs = np.arange(1,len(sign_vars)+1,1)
plt.figure(figsize=(10, 10))
plt.errorbar(indexs,coefs[sign_vars],yerr=ses[sign_vars]*1.96,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=2)
plt.axhline(y=0, color='gray', linestyle='-')
plt.xticks(range(1, len(sign_vars) + 1), data_frame.columns[sign_vars],rotation='vertical')
plt.show()

indexs = np.arange(1,dim+1,1)
plt.figure(figsize=(10, 10))
plt.errorbar(indexs,coefs,yerr=ses*1.96,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=2)
plt.axhline(y=0, color='gray', linestyle='-')
plt.xticks(range(1, dim + 1), data_frame.columns[:-2],rotation='vertical')
plt.show()
