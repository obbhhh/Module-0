"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
import math

# 1. Multiplies two numbers
def mul(a: float, b: float): # type: ignore
    return a * b

# 2. Returns the input unchanged
def id(x: float):
    return x

# 3. Adds two numbers
def add(a: float, b: float):
    return a + b

# 4. Negates a number
def neg(x: float):
    return -x

# 5. Checks if one number is less than another
def lt(a: float, b: float):
    return a < b

# 6. Checks if two numbers are equal
def eq(a: float, b: float):
    return a == b

# 7. Returns the larger of two numbers
def max(a: float, b: float):
    return a if a > b else b

# 8. Checks if two numbers are close in value
def is_close(a: float, b: float, tol: float=1e-5):
    return abs(a - b) < tol

# 9. Calculates the sigmoid function
def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))

# 10. Applies the ReLU activation function
def relu(x: float):
    return max(0, x)

# 11. Calculates the natural logarithm
def log(x: float):
    return math.log(x)

# 12. Calculates the exponential function
def exp(x: float):
    return math.exp(x)

# 13. Calculates the reciprocal
def inv(x: float):
    return 1 / x

# 14. Computes the derivative of log times a second arg
def log_back(x: float, dout: float):
    return dout / x

# 15. Computes the derivative of reciprocal times a second arg
def inv_back(x: float, dout: float):
    return -dout / (x ** 2)

# 16. Computes the derivative of ReLU times a second arg
def relu_back(x: float, dout: float):
    return dout if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(fn: Callable, iterable: Iterable):
    """
    将函数 `fn` 应用到 `iterable` 的每个元素上，返回结果列表。

    参数:
        fn (callable): 接受一个参数的函数。
        iterable (iterable): 可迭代对象。

    返回:
        list: 应用 `fn` 后的结果列表。
    """
    return [fn(x) for x in iterable]

def zipWith(fn: Callable, iterable1: Iterable, iterable2: Iterable):
    """
    使用函数 `fn` 将 `iterable1` 和 `iterable2` 的对应元素组合起来，返回结果列表。

    参数:
        fn (callable): 接受两个参数的函数。
        iterable1 (iterable): 第一个可迭代对象。
        iterable2 (iterable): 第二个可迭代对象。

    返回:
        list: 组合后的结果列表。
    """
    return [fn(x, y) for x, y in zip(iterable1, iterable2)]

def reduce(fn: Callable, iterable: Iterable, initial: Any=None):
    """
    使用函数 `fn` 将 `iterable` 缩减为单个值。

    参数:
        fn (callable): 接受两个参数的函数。
        iterable (iterable): 可迭代对象。
        initial (optional): 初始值。如果未提供，则使用 `iterable` 的第一个元素。

    返回:
        缩减后的值。
    """
    it = iter(iterable)
    if initial is None:
        try:
            initial = next(it)
        except StopIteration:
            raise ValueError("reduce() of empty iterable with no initial value")
    result = initial
    for x in it:    
        result = fn(result, x)
    return result


def addLists(list1: List[float], list2: List[float]) -> List[float]:
    return [a + b for a, b in zip(list1, list2)]

def negList(lst: List[float]) -> List[float]:
    return [-x for x in lst]

def prod(lst: List[float]) -> float:
    result = 1.0
    for x in lst:
        result *= x
    return result

def sum(lst: List[float]) -> float:
    res = 0.0
    for x in lst:
        res += x
    return res

# TODO: Implement for Task 0.3.
