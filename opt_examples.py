from pyautodiff.optimizer import Optimizer
import numpy as np
from sklearn.datasets import make_regression, make_classification


# Creates a regression problem with 3 features
X,y,coef = make_regression(n_features=3,n_informative=3,coef=True)

# User defined loss function
def user_mse(y,y_preds):
    return (y-y_preds)**2


opt = Optimizer(loss=user_mse,optimizer='gd',regularizer='ridge',lam=0.01)
opt.fit(X,y,iters=1000)

print('Actual Values:')
print('======')
print(coef)

print('Computed Values:')
print('======')
print(opt.coefs)
