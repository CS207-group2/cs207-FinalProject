from pyautodiff.optimizer import Optimizer
import pyautodiff.admath as admath
import numpy as np
from sklearn.datasets import make_regression, make_classification


# # Creates a regression problem with 3 features
# X,y,coef = make_regression(n_features=3,n_informative=3,coef=True)
#
# # User defined loss function
# def user_mse(y,y_preds):
#     return (y-y_preds)**2
#
#
# opt = Optimizer(loss=user_mse,optimizer='sgd',regularizer='ridge',lam=0.01)
# opt.fit(X,y,iters=1000)
#
# print('Actual Values:')
# print('======')
# print(coef)
#
# print('Computed Values:')
# print('======')
# print(opt.coefs)

# Creates a classification problem with 3 features
X2,y2 = make_classification(n_features=3,n_informative=3,n_redundant=0)

def user_cross_entropy(y,y_preds):
    if y == 1:
        return -admath.log(y_preds)
    else:
        return -admath.log(1-y_preds)


opt = Optimizer(loss=user_cross_entropy,optimizer='sgd',regularizer='ridge',lam=0.01,problem_type='classification')
opt.fit(X2,y2,iters=2500)
y_preds = opt.predict(X2)

print('Computed Values:')
print('======')
print(opt.coefs)
print(opt.score(X2,y2))
