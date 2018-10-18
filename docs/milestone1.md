# Introduction
This package can automatically differentiate a function input into the program. By applying the chain rule repeatedly to these operations, derivatives of arbitrary order can be computed automatically, accurately to machine accuracy. It currently supports forward-mode differentiation.

The major application of automatic differentiation is gradient-based optimization, which is commonly used as the foundation of neural nets.

# Background  
Automatic differentiation breaks down any function into its elementary functions using a graph structure and calculates the derivative while retaining the function by using dual numbers. This is accomplished by substituting (x + ɛ x-prime) for x in f(x).

As the steps of the graph structure become successively more complex, the derivatives of the preceding steps are used to compute the derivatives.

The chain rule is important for increasing the robustness of the automatic differentiation class, especially because it allows for the class to calculate the derivative of compositions (which are an important part of approximating non-linear functions).

# How to Use *AutoDiff*
The user can use AutoDiff by passing a function to the AutoDiff constructor to create an AutoDiff object. Then, the user can evaluate the derivative of that function at a certain value by passing in that value to the object. This object can then be called to return the derivative of the function evaluated at that point.

Scalar function case:

```python
>>> import AutoDiff
>>> def square_fn(x):
...	return x ** 2
>>> ad_square = AutoDiff(square_fn)
>>> ad_square.get_der(3)
6
```

Vector function case:
```python
>>> import AutoDiff
>>> def square_fn(x):
...	return x ** 2
>>> ad_square = AutoDiff(square_fn)
>>> ad_square.get_der([1,2])
np.array([2,4])
```

In cases where the user wants to use operations such as sin/cos, they should call those functions from the AutoDiff library so that the derivative can be automatically computed.

SINE, COSINE, EXPONENTIAL function case:
```python
>>> import AutoDiff
>>> def sin_fn(x):
...	return Autodiff.sin(x)
>>> ad_sin = AutoDiff(sin_fn)
>>> ad_sin.get_der(0)
1
```

# Software Organization
Discuss how you plan on organizing your software package.
* What will the directory structure look like?

<pre>
AutoDiff\
         AutoDiff\
               __init__.py
               AutoDiff.py
               tests\
                    	__init__.py
	   	Dual\
			Dual.py
			__init__.py
	   	math\
			math.py
			__init__.py	
         README.md
         setup.py
         LICENSE

</pre>

* What modules do you plan on including?  What is their basic functionality?
	Numpy: to do vector operations simultaneously
Math: to do sin/cos/exp functions

* Where will your test suite live?  Will you use `TravisCI`? `Coveralls`?
	The github page of our package will include TravisCI and Coveralls.
* How will you distribute your package (e.g. `PyPI`)?
Yes, we will distribute the package via PyPI.

# Implementation
Discuss how you plan on implementing the forward mode of automatic differentiation.
* What are the core data structures?
Dual object (from AutoDiff.dual)
* What classes will you implement?
AutoDiff.math
AutoDiff.dual
AutoDiff.array (will be defined if needed)
* What method and name attributes will your classes have?
AutoDiff.math
Methods:
Sin, cos, cot, arcsin, exp, log10, loge, ln etc
		These methods will be compatible with dual numbers, scalars, as well as vectors.
AutoDiff.dual
	Methods:
dunder methods (add,radd)
	Attributes:
Value
Derivative

* What external dependencies will you rely on?
numpy
math
* How will you deal with elementary functions like `sin` and `exp`?
We are going to import these functions from math and numpy packages and wrap it under our package to account for differentiation operations

Be sure to consider a variety of use cases.  For example, don't limit your design to scalar
functions of scalar values.  Make sure you can handle the situations of vector functions of vectors and scalar functions of vectors.  Don't forget that people will want to use your library in algorithms like Newton's method (among others).
