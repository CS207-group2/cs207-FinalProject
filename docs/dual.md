# autodiff.dual

## Implementation of unary operations
We implemented the following unary operations:
``__neg__`` and ``__pos__``.

## Implementation of binary operations

We provide full implementations off all binary differentiable operations with dual and non-dual numbers. When we want to operate a with u, thus we have 4 possibities: 

1. a and u are both scalars
2. a is a dual number, b is a scalar
3. a is a scalar, b is a dual number
4. a and u are both dula numbers

Operations 1 is already implemented by Python, hence we only need to implement operations 2 through 4.

In the class ``autodiff.dual`` we will implement the following binary operations:
* Addition
* Substration
* Multiplication
* Division
* Power

Below are the derivative formula for each operation. Dual numbers are marked as a function with a variable, e.g. u(x) or a(x) with their derivatives, a' and u'. Scalars are simply labeled without variable: a and b. 

### Addition operation
Overload `__add__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%28x%29%20&plus;%20u%5CBig%5D%3D%20a%5E%5Cprime)

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%28x%29%20&plus;%20u%28x%29%5CBig%5D%3D%20a%5E%5Cprime%20&plus;%20u%5E%5Cprime)

Overload `__radd__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%20&plus;%20u%28x%29%5CBig%5D%3D%20u%5E%5Cprime)

### Substration operation
Overload `__sub__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%28x%29%20-%20u%5CBig%5D%3D%20a%5E%5Cprime)

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%28x%29%20-%20u%28x%29%5CBig%5D%3D%20a%5E%5Cprime%20-%20u%5E%5Cprime)

Overload `__rsub__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%20-%20u%28x%29%5CBig%5D%3D%20-%20u%5E%5Cprime)


### Multiplication operation
Overload `__mul__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%28x%29u%5CBig%5D%3D%20u%20a%5E%5Cprime)

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Ba%28x%29u%28x%29%5CBig%5D%3D%20a%5E%5Cprime%20u%20&plus;%20a%20u%5E%5Cprime)

Overload `__rmul__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5Bau%28x%29%5CBig%5D%3D%20a%20u%5E%5Cprime)


### Division operation
Overload `__truediv__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5B%5Cdfrac%7Ba%28x%29%7D%7Bu%7D%5CBig%5D%3D%20%5Cdfrac%7Ba%5E%5Cprime%7D%7Bu%7D)

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5B%5Cdfrac%7Ba%28x%29%7D%7Bu%28x%29%7D%5CBig%5D%3D%20-%5Cdfrac%7Ba%5E%5Cprime%20u%20-%20a%20u%5E%5Cprime%7D%7B%7Bu%28x%29%7D%5E2%7D)

Overload `__rtruediv__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%5CBig%5B%5Cdfrac%7Ba%7D%7Bu%28x%29%7D%5CBig%5D%3D%20-%5Cdfrac%7Ba%7D%7B%7Bu%28x%29%7D%5E2%7D%20u%5E%5Cprime)


### Power operation
Overload `__pow__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%20a%28x%29%5En%20%3D%20n%20a%5E%7Bn-1%7D%20a%5E%7B%5Cprime%7D)

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%20a%28x%29%5E%7Bu%28x%29%7D%20%3D%20a%5E%7Bu-1%7D%20%5CBig%5Bu%20a%5E%5Cprime%20&plus;%20a%20%5Clog%28a%29%20u%5E%5Cprime%20%5CBig%5D)

Overload `__rpow__` to account for:

![equation](http://latex.codecogs.com/gif.latex?%5Cdfrac%7Bd%7D%7Bdx%7D%20a%5E%7Bu%28x%29%7D%20%3D%20log%28a%29%20a%5E%7Bu%28x%29%7D%20u%5E%5Cprime)
