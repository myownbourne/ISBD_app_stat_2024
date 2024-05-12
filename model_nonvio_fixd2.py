import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from doubleml import DoubleMLData,DoubleMLPLR
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

np.random.seed(3141)

data_frame = pd.read_csv("crime_vio.csv")
cols_nonvio = np.loadtxt('cols_nonvio.txt', dtype=str)

#del some columns
#del_cols = ['agePct12t29','agePct16t24','numbUrban','OwnOccHiQuart','RentHighQ']
data_frame2 = data_frame[cols_nonvio]
data = data_frame2.values

n, p = data.shape
XD = data[:,:-2]
y1 = data[:,-2]
y2 = data[:,-1]
y1 = y1/1000
y2 = y2/1000

ss = StandardScaler()
XD_s = ss.fit_transform(XD)

d_name = 'racepctblack'
d_index = np.where(cols_nonvio == d_name)[0][0]
d = XD_s[:,d_index]
X = np.delete(XD_s, d_index, axis=1)

dml_data = DoubleMLData.from_arrays(X, y2, d)
learner_ls = LassoCV(max_iter=20000)
learner_for = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=20, min_samples_leaf=2)
learner_SVR = SVR(C=1.0, epsilon=0.2)
ml_l = clone(learner_for)
ml_m = clone(learner_for)
#ml_g = clone(learner)

dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)
dml_plr.fit()
print(dml_plr)

dml_plr.sensitivity_analysis(cf_y=0.01, cf_d=0.01)

print(dml_plr.sensitivity_summary)




reg_ls1 = LassoCV(max_iter=20000).fit(X, d)
non_zero_index1 = np.nonzero(reg_ls1.coef_)[0]
reg_ols1 = LinearRegression().fit(X[:,non_zero_index1], d)
d_hat = reg_ols1.predict(X[:,non_zero_index1])
v_hat = d - d_hat

plt.hist(v_hat, bins=40, density=True, alpha=0.6, color='g', edgecolor='black')
mu1, std1 = norm.fit(v_hat)
x_nor1 = np.linspace(min(v_hat), max(v_hat), 100)
y_nor1 = norm.pdf(x_nor1, mu1, std1)
plt.plot(x_nor1, y_nor1, 'r-', linewidth=2)
plt.show()

plt.scatter(np.arange(1,1902,1),v_hat,s=2)
plt.show()

reg_ls = LassoCV(max_iter=20000).fit(X, y2)
non_zero_index = np.nonzero(reg_ls.coef_)[0]
reg_ols = LinearRegression().fit(X[:,non_zero_index], y2)
y_hat = reg_ols.predict(X[:,non_zero_index])
eps_hat = y2 - y_hat


plt.hist(eps_hat, bins=40, density=True, alpha=0.6, color='g', edgecolor='black')
mu, std = norm.fit(eps_hat)
x_nor = np.linspace(min(eps_hat), max(eps_hat), 100)
y_nor = norm.pdf(x_nor, mu, std)
plt.plot(x_nor, y_nor, 'r-', linewidth=2)
plt.show()


plt.scatter(np.arange(1,1902,1),eps_hat,s=2)
plt.show()

