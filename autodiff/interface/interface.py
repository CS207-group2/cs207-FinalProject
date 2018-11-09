from autodiff.dual.dual import Dual
import inspect

class AutoDiff:
    def __init__(self, fn, ndim=1):
        """
        fn : function, the function of which we want to calculate the derivative
        ndim : float, the number of dimensions of the function
        l : the number of parameters in the function
            (e.g. in lambda x, y: x**2 + y**2) it would be x,y (so l = 2)
        """
        self.fn = fn
        self.ndim = ndim
        sig = inspect.signature(self.fn)
        self.l = len(list(sig.parameters))

    def get_der(self, val):
        """ Returns derivatives of the function evaluated at values given.

        INPUTS
        =======
        val : single number, a list of numbers, or a list of lists

        RETURNS
        =======
        Derivates in the same shape given

        EXAMPLE
        =======
        >>> a = AutoDiff(lambda x,y: 5*x + 4*y)
        >>> a.get_der([[6.7, 4],[2,3],[4.5,6]])
        [[5, 4], [5, 4], [5, 4]]
        """
        ders = []
        if self.ndim >1:
            for i in range(self.ndim):
                def fxn(*args):
                    return self.fn(*args)[i]
                a = AutoDiff(fxn,ndim=1)
                a.l=self.l
                ders.append(a.get_der(val))
            return ders
        else:
            if self.l >= 2:

                #for list of lists, each list evaluated at different variables
                if any(isinstance(el, list) for el in val):
                    list_der = []
                    for p in val:
                        list_der.append(self.get_der(p))
                    return list_der
                elif self.l != len(val):
                    raise Exception('Function requires {} values that correspond to the multiple variables'.format(self.l))
                else:
                    #for a list of numbers, evaluated at different variables.
                    for i in range(self.l):
                        new_val = val.copy()
                        new_val[i] = Dual(new_val[i])
                        v = self.fn(*new_val)
                        #Check if variable is in the function. (E.g., function paramaters are x, y and function is x.)
                        if type(v) is Dual:
                            ders.append(self.fn(*new_val).der)
                        else:
                            ders.append(0)
                    return ders
            #for a list of numbers, evaluated at a single variable.
            if (isinstance(val,list)):
                for v in val:
                    a = Dual(v)
                    ders.append(self.fn(a).der)
                return ders
            else:
                a = Dual(val)
                return self.fn(a).der
