from autodiff.interface.interface import AutoDiff as AD
import autodiff.admath.admath as admath
import numpy as np
# from sklearn.metrics import accuracy_score
import random

class Optimizer:
    def __init__(self,lr=0.01,loss='mse',optimizer='gd',regularizer=None,lam=None):
        """
        lr : float, the learning rate for our algorithm
        loss : string or function, a string that indicates which of the pre-specified
            loss functions to use or a function representing the loss
        optimizer : string, indicates which optimizer to use. Supported optimizers are 'gd'
            for gradient descent and 'sgd' for stochastic gradient descent
        regularizer : string, indicates which regularizer to use. Supported regularizers are
            'l1' for lasso and 'l2' for ridge
        lam : float, the regularization parameter value
        """
        self.lr = lr

        self.loss_fn = self._get_loss_fn(loss)
        self.cost = self._get_cost_fn(self.loss_fn,optimizer)
        self.optimizer = self._get_optimizer(optimizer)
        self.regularizer = regularizer
        self.lam = lam

    def _get_loss_fn(self,loss):
        """ Convert loss into a function if it was inputted as a string
        """
        if type(loss) == str:
            if loss == 'mse':
                return self.mse
            else:
                raise Exception('Invalid Loss')
        else: # user-inputted loss function
            return loss

    def mse(self,y,y_preds):
        return (y-y_preds)**2

    def _get_cost_fn(self,loss_fn,optimizer):
        """ Return a cost function given the loss and optimizer
        """
        if optimizer == 'gd':
            return self.cost_fn_gd
        elif optimizer == 'sgd':
            return self.cost_fn_sgd

    def cost_fn_gd(self,X,Y):
        """ Uses the loss function saved in self.loss_fn to create a cost function for gd
        """
        def func(*args):
            betas = np.array(args)
            losses = []
            for x,y in zip(X,Y):
                y_preds = 0
                for x_i,b_i in zip(x,betas):
                    y_preds += x_i*b_i
                losses.append(self.loss_fn(y,y_preds))
            # This is the squared loss term
            base_val = admath.sum(losses)/len(Y)

            # This adds the regularizer loss
            if self.regularizer is not None:
                reg_loss = self._compute_reg_loss(betas)
                return base_val+reg_loss
            else:
                return base_val
        return func

    def cost_fn_sgd(self,X_i,Y_i):
        """ Uses the loss function saved in self.loss_fn to create a cost function for sgd
        """
        def func(*args):
            betas = np.array(args)
            y_preds = 0
            for x_i,b_i in zip(X_i,betas):
                y_preds += x_i*b_i

            # This is the squared loss term
            base_val = self.loss_fn(Y_i,y_preds)

            # This adds the regularizer loss
            if self.regularizer is not None:
                reg_loss = self._compute_reg_loss(betas)
                return base_val+reg_loss
            else:
                return base_val
        return func

    def _get_optimizer(self,optimizer):
        """ Convert optimizer string into a function
        """
        if type(optimizer) == str:
            if optimizer == 'gd':
                return self.fit_gd
            elif optimizer == 'sgd':
                return self.fit_sgd
            else:
                raise Exception('Invalid optimizer')
#         else: # user-inputted optimizer
#             return optimizer


    def predict(self, X):
        return X@self.coefs

    def _compute_reg_loss(self,betas):
        """ Calculates the regularization loss term
        """

        reg = []
        eps=1e-4
        for w in betas:
            if self.regularizer == 'lasso':
                reg.append(admath.abs(w+eps)*self.lam)
            elif self.regularizer == 'ridge':
                reg.append(w**2*self.lam)
        return admath.sum(reg)


    def mse_loss(self,y,y_preds):
        return np.sum((y-y_preds)**2)/len(y)


    def fit(self, X, y, iters=1000, verbose=False):
        """ Wrapper method for fit. Calls the function saved in self.optimizer which is
        fit_gd or fit_sgd depending on whether the optimizer is gd or sgd.
        """
        self.optimizer(X,y,iters,verbose=verbose)

    def fit_gd(self, X, y, iters=1000,verbose=False):
        self.coefs = np.random.rand(X.shape[1])
        cur_loss = self.mse_loss(y, self.predict(X))
        i = 0
        while i < iters:
            func_for_opt = self.cost(X,y)
            ad_obj = AD(func_for_opt,multivar=True)
            if verbose:
                print("=====\nIter {} Loss: {}".format(i,cur_loss))
                print(self.coefs)
            grads = ad_obj.get_der(list(self.coefs))
            for idx,grad in enumerate(grads):
                if verbose:
                    print('coef{}: {}'.format(idx,self.coefs[idx]))
                self.coefs[idx] = self.coefs[idx] - self.lr*grad

            cur_loss = self.mse_loss(y, self.predict(X))
            i+=1


    def fit_sgd(self, X, y, iters=1000,verbose=False):
        self.coefs = np.random.rand(X.shape[1])
        cur_loss = self.mse_loss(y, self.predict(X))
        i = 0
        while i < iters:
            sgd_index = random.randint(0,X.shape[0]-1)
            func_for_opt = self.cost(X[sgd_index],y[sgd_index])
            ad_obj = AD(func_for_opt,multivar=True)
            if verbose:
                print("=====\nIter {} Loss: {}".format(i,cur_loss))
                print(self.coefs)
            grads = ad_obj.get_der(list(self.coefs))

            for idx,grad in enumerate(grads):
                if verbose:
                    print('coef{}: {}'.format(idx,self.coefs[idx]))
                    print(grad)
                self.coefs[idx] = self.coefs[idx] - self.lr*grad

            cur_loss = self.mse_loss(y, self.predict(X))
            i+=1
