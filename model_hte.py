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
data_frame2 = data_frame[cols_nonvio]


data_frame2_c = data_frame2.copy()
X_cols = data_frame2_c.columns[:-2]
X_data = data_frame2_c[X_cols].values
scaler = StandardScaler()
X_data_nor= scaler.fit_transform(X_data)
data_frame2_c.loc[:, X_cols] = X_data_nor
data_frame2 = data_frame2_c
data_frame2.iloc[:, -1] = data_frame2.iloc[:, -1] / 1000

data_frame2 = data_frame2.drop('ViolentCrimesPerPop', axis=1)

dml_data = DoubleMLData(data_frame2, y_col='nonViolPerPop',d_cols='PctKids2Par')

learner_ls = LassoCV(max_iter=20000)
ml_l = clone(learner_ls)
ml_m = clone(learner_ls)
dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m)
dml_plr.fit()

import patsy
design_matrix = patsy.dmatrix("bs(x, df=7, degree=4)", {"x": data_frame2["MalePctDivorce"]})
spline_basis = pd.DataFrame(design_matrix)


cate = dml_plr.cate(spline_basis)
print(cate)

new_data = {"x": np.linspace(-1, 1, 100)}
spline_grid = pd.DataFrame(patsy.build_design_matrices([design_matrix.design_info], new_data)[0])
df_cate = cate.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)
print(df_cate)


df_cate['x'] = new_data['x']
fig, ax = plt.subplots()
ax.plot(df_cate['x'],df_cate['effect'], label='Estimated Effect')
ax.fill_between(df_cate['x'], df_cate['2.5 %'], df_cate['97.5 %'], color='b', alpha=.3, label='Confidence Interval')

plt.legend()
plt.title('CATE (treatment variable - PctKids2Par)')
plt.xlabel('MalePctDivorce')
_ =  plt.ylabel('Effect and 95%-CI')
plt.show()