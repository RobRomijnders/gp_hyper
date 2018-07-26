import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF
from scipy.stats import norm, multivariate_normal

from utils.util import pandas_to_numpy

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# print(train.head())

# Some Matplotlib magic
# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
# prices.hist()
# plt.show()

# Make a train - val split
X_train, X_val, y_train, y_val = pandas_to_numpy(train, test_size=0.2)
num_train, num_features = X_train.shape

# Explore what would be the default log likelihood
normal = norm(loc=np.mean(y_val), scale=np.std(y_val))
default_log_like = np.sum(normal.logpdf(y_val))
print(f'Default log likelihood is {default_log_like}')

"""Search the hyperparameter space for ourselves"""
l_range = np.linspace(0.01, 0.3, num=10)  # Length scale parameter
c_range = np.linspace(-.5, 2., num=5)  # Covariance scale parameter
L_mesh, C_mesh = np.meshgrid(l_range, c_range)


def loop_over_mesh():
    # Loop over all parameter settings in the mesh and evaluate the log likelihood
    for l, c in zip(L_mesh.flatten(), C_mesh.flatten()):
        kernel = 10**c * RBF(length_scale=l*num_features)
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.1,
                                      normalize_y=True,
                                      optimizer=None)
        gp.fit(X_train, y_train)
        # yield gp.log_marginal_likelihood()  # Use this for training log likelihood
        # Yield the log likelihood on the validation set
        yield multivariate_normal(*gp.predict(X_val, return_cov=True)).logpdf(y_val)


log_like = np.reshape(np.array(list(loop_over_mesh())), newshape=L_mesh.shape)

best_log_like = np.max(log_like)
print(f'Maximum log likelihood is {best_log_like:.2f}')

# Again some matplotlib magic
plt.figure()
CS = plt.contour(L_mesh, C_mesh, log_like, levels=np.linspace(default_log_like, best_log_like, 15))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('length scale')
plt.ylabel('covariance scale')
plt.title('Simplest default with labels')
plt.show()

# """Or optimize over it"""
#
# kernel = 1.0*RBF(length_scale_bounds=[num_features*0.01, num_features*4])
# gp = GaussianProcessRegressor(kernel=kernel,
#                               alpha=0.1,
#                               n_restarts_optimizer=0)
# gp.fit(X_train, y_train)
# print(f'Learned kernel is {gp.kernel_}')
#
# print(f'Log likelihood for the validation data is {gp.log_marginal_likelihood()}')
# y_val_pred = gp.predict(X_val)

# for pred, target in zip(y_val_pred, y_val):
#     print(pred, target)
