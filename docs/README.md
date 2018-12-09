# Content
- [Introduction](#Introduction)
- [Usage](#Usage)
  - [Installation](#Installation)
  - [How to use](#How-to-use-Autodiff)
- [Background](#Background)
- [Software organization](#Software-Organization)
- [Implementation Details](#Implementation-Details)
  - [Interface](#Interface-Class)
  - [Dual](#Dual-Class)
  - [Admath](#Admath-Module)
- [External Dependencies](#External-Dependencies)
- [Future Implementations](#Future-Implementations)

# Introduction
Automatic differentiation is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. Automatic differentiation breaks down a function by looking at the sequence of elementary arithmetic operations (addition, subtraction, multiplication and division) and elementary functions (exponential, log10, log2, loge, sin, cos, etc). By applying the chain rule repeatedly to these operations, derivatives of arbitrary order can be computed automatically, accurately to machine accuracy. A major application of automatic differentiation is gradient-based optimization, such as gradient descent, which is commonly used as the foundation of many machine learning algorithms including Neural Networks.

This package, `pyautodiff`, is a package of automatic differentiation, which means it can automatically differentiate a function input into the program.

The package currently supports forward-mode differentiation, which means the chain rule is traversed from inside to outside.

# Usage

## Installation
* The package is available on `PyPI`
```
pip install pyautodiff
```

## How to use *pyautodiff*?
The user can use pyautodiff by passing a function to the AutoDiff constructor to create an AutoDiff object. Then, the user can evaluate the derivative and output value of that function at a certain value by passing in that input value to the object. This object can then be called to return the derivative and the output values of the function evaluated at that point.

Scalar function case:
```python
>>> from pyautodiff.interface import AutoDiff as AD
>>> def square_fn(x):
...    return x ** 2
>>> ad_square = AD(square_fn)
>>> ad_square.get_der(3)
6
>>> ad_square.get_val(3)
9
```

Vector function case:
```python
>>> from pyautodiff.interface import AutoDiff as AD
>>> def square_fn(x):
...    return x ** 2
>>> ad_square = AD(square_fn)
>>> ad_square.get_der([1,2])
[2, 4]
>>> ad_square.get_val([1,2])
[1, 4]
```

In cases where the user wants to use operations such as sin/cos, they should call those functions from the pyautodiff library so that the derivative can be automatically computed.

SINE, COSINE, EXPONENTIAL function case:
```python
>>> from pyautodiff.interface import AutoDiff as AD
>>> import pyautodiff.admath as admath

>>> def sin_fn(x):
...    return 5*admath.sin(x)
>>> ad_sin = AutoDiff(sin_fn)
>>> ad_sin.get_der(0)
5
>>> ad_sin.get_val(0)
0
```

Multivariable case (one function):
```python
>>> from pyautodiff.interface import AutoDiff as AD

>>> def my_fn_1d(x, y):
...	  return x**2 + y**2
>>> fn = AD(my_fn_1d)
>>> fn.get_der([[1,2],[3,4],[5,6]])
[[2, 4], [6, 8], [10, 12]]
>>> fn.get_val([[1,2],[3,4],[5,6]])
[5, 25, 61]
```


Multivariable case (multiple functions):
```python
>>> from pyautodiff.interface import AutoDiff as AD

>>> def my_fn_2d(x, y):
...	  return [x**2 + y**2, x + 2+y]
>>> fn = AD(my_fn_2d, ndim=2)
>>> fn.get_der([1,2])
[[2, 4], [1, 1]]
>>> fn.get_val([1,2])
[5, 5]
```


# Background
Automatic differentiation breaks down any function into its elementary functions using a graph structure, where every node is an operation, and calculates the derivative on top of the numerical value. The simultaneous value and derivative calculation is accomplished by using dual numbers, which are numbers have an additional component ɛ on top of its real component (called dual component).

Dual numbers can simply be used by substituting (x + ɛ x') for x in f(x) where f can be any one operation. The important idea is that after every mathematical operation, the real part will represent the numerical value of the expression and the dual part will reflect the derivative. This property makes it very convenient for derivative calculations of heavily nested functions because of the chain rule in derivative calculation which states that the derivative of f(g(x)) is f'(g(x)) * g'(x). Since the derivative of a nested function relies on both the value and the derivative of the inner function, we can see that the automatic storage of both the value and derivative after every operation is very convenient for this task.



# Software organization
- High-level overview of how the software is organized.
  * Directory structure
  ```
   FinalProject\
         pyautodiff\
               __init__.py
               admath.py
               dual.py
               interface.py
               optimizier.py
         test\
              __init__.py
              .coverage
              test_admath.py
              test_dual.py
              test_interface.py
              test_optimizer.py
         docs\
              dual.md
              README.md
         README.md
         setup.py
         LICENSE.txt
         .gitignore
         .travis.yml
         setup.cfg
  ```
  * Modules and classes
    * `admath`
      * this is the module for math computation. It leverages numpy library and provides functions including elementary functions (exponential, log10, log2, loge, sin, cos)
    * `interface`
      * this is the main class where an instance of our class can be instantiated by passing in a function.
    * `dual`
      * this is the class for dual number input, which contains the `val` and `der` attributes that stores the numerical value and derivate respectively.
    * `optimizer`
      * this is a wrapper for pyautodiff with an API similar to sklearn for fitting the data dependent and independent variables.
      * it calcualtes gradients and performs gradient descent.
  * Test
    * Tests of this package are in the `test` folder.
    * They are run by `TravisCI` and the coverage is examined by `Coveralls`
    * We have embedded the badges in the README of the package
  * How can someone install your package?
    * The package is available on `PyPI`
    ```
    pip install pyautodiff
    ```

# Implementation details
Currently, the pyautodiff package contains 2 classes and 2 modules.

### Interface Class
#### Usage
Interface is our main class where an instance of our class can be instantiated by passing in a function. Next, the user can pass in a scalar or a list of numbers into the get_der method to evaluate the derivative(s) of the function with respect to the point(s). Furthermore, our class supports multivariable differentiation, where the user can write a multivariable function, pass in a 2d list where each list represents the derivative calculation at each value, and get back the Jacobian matrix.

#### Implementation
The interface object contains 3 attributes which are fn, ndim, and l. fn represents the function that we want to evaluate a derivative at, ndim is the number of dimensions of the function (how many functions we want to evaluate), and l is the number of variables in the function. While fn is passed into the function and ndim is optionally passed in, l is inferred from fn through the usage of the inspect module.

The get_der function works by first determining which type of an input is given to the function through the usage of ndim and l as well as what type of an argument is passed into the get_der function (whether it's a scalar or a list). The function then handles these cases separately. For the single variable case (l = 1), if function argument is a scalar, then a dual object is instantiated and passed into fn, the function attribute, so that the derivative can be calculated. If the argument is a list, then the same operation done in a scalar is repeatedly done through a for loop and the result will be appended to a list and returned. In the multivariable case (l>1), then the derivative with respect to each variable is calculated separately and appended to the returned list.


### Dual Class
#### Usage
The dual class represents a dual number. The user does not have to explicitly instantiate it to use our package as it will be instantiated by the Interface class automatically.

#### Implementation
The dual object contains val and der attributes, representing the numerical value and derivative respectively. It contains dunder methods to handle all basic math operations such as add, multiply, power, in cases where both numbers being added are dual numbers (eg a+b where both a and b are dual) as well as in cases where the left or the right side of the expression is a scalar (eg a+b where a is scalar and b is dual).

Additional details of the dual class can be found [here](dual.md).

### Admath Module
#### Usage
This admath module performs both value and derivative calculations of elemental functions, such as trig and exponential, for the dual class. We use the same function names as numpy to allow for easier usability for people already used to numpy. The functions in this module also work with scalars.

#### Implementation
We implemented the following functions:
1. Trigonometric functions
  - sin(x)
  - cos(x)
  - tan(x)
  - arcsin(x)
  - arcos(x)
  - arctan(x)
2. Logarithmic functions
  - log(x)
  - log10(x)
  - log2(x)
  - logb(x, base)

3. Hyperbolic functions
  - sinh(x)
  - cosh(x)
  - tanh(x)

4. Miscellaneous functions
  - exp(x)
  - sqrt(x)
  - logistic(x)
  - power(expo, base)
  - sum(x)
  - abs(x)

In cases where x is a scalar, we simply return the numpy equivalent (eg np.sin(x)). When x is dual, we manually set val and der of the dual object. We set the der by figuring out symbolically what the derivative should be (sin(x) should be cos(x)) and applying the chain rule (multiplying x.der to cos(x)). This way, our program can automatically apply the chain rule to our inputs and handle nested functions with ease. Again, like the scalar case, we use numpy to do the actual elemental calculations.

### Optimizer Module
#### Usage
This optimizer module is a wrapper for pyautodiff with an API similar to sklearn for fitting dependent and independent variables in the data

It leverages autodiff to calculate the gradients and performs gradient descent. It can take in a custom loss function, regularizer, and/or optimizer.

#### Implementation

# External Dependencies
- numpy: to perform calculations on elemental functions such as sin and exponent


# Future Implementations
- Newton's method
- Visualization of the value and derivative at each step of the forward mode
- Reverse mode
