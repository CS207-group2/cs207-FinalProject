import pytest
from autodiff.interface.interface import AutoDiff as AD
import autodiff.admath.admath as admath

import numpy as np


def my_fn_2d(x, y):
	return [x**2 + y**2, x + 2+y]

def my_fn_1d(x,y):
	return x**2 + y**2


def test_list():
	fn = AD(my_fn_1d)
	der = fn.get_der([1,2])
	assert der == [2, 4]

def test_list_lists():
	fn = AD(my_fn_1d)
	der = fn.get_der([[1,2],[3,4],[5,6]])
	assert der == [[2, 4], [6, 8], [10, 12]], [[1, 1], [1, 1], [1, 1]]


def test_2d_fn():
	fn = AD(my_fn_2d, ndim=2)
	der = fn.get_der([1,2])
	assert der == [[2, 4], [1, 1]]

def test_func_w_mult_params_single_var():
	fn = lambda x,y: x**2
	ad_fn = AD(fn)
	der = ad_fn.get_der([1,2])
	assert der == [2, 0]

def square_fn(x):
        return x ** 2

def test_square_fn():
	ad_square = AD(square_fn)
	der = ad_square.get_der(3)
	assert der == 6
	der1 = ad_square.get_der([1,2,3,4])
	assert der1 == [2, 4, 6, 8]

def test_get_der_types():
	with pytest.raises(TypeError):
        	AD.get_der('string')
	with pytest.raises(TypeError):
        	AD.get_der(dict[1:'a', 2:'b'])

def test_get_der_lenlist():
	a = AD(lambda x,y: 3*x**2 + 2*y**3)
	with pytest.raises(Exception):
        	a.get_der(1, 2, 3)
	with pytest.raises(Exception):
			a.get_der([1, 2, 3], [3, 4, 5], [1, 3, 4])

def test_exception():
	fn = AD(my_fn_2d, ndim=2)
	with pytest.raises(Exception):
		fn.get_der([1,2,3])

def my_fn_cos(x):
	return 5*admath.cos(x)

def my_fn_nested_1(x):
	return 5*x**2 * 2*admath.cos(x)

def test_cos():
	cos_fn = AD(my_fn_cos)
	assert cos_fn.get_der(5) == -5*np.sin(5)

def test_nested_1():
	nested_fn = AD(my_fn_nested_1)
	xs = [-10,2,5,10]
	for x in xs:
		assert nested_fn.get_der(x) == pytest.approx(-10*x*(x*np.sin(x)-2*np.cos(x)))

def my_fn_nested_2(x):
	return 5*admath.log(admath.sin(x))

def test_nested_2():
	nested_fn = AD(my_fn_nested_2)
	xs = [0.4, 0.9, 1.2]
	for x in xs:
		assert nested_fn.get_der(x) == pytest.approx(5/np.tan(x))
