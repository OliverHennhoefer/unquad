import numpy as np
import pandas as pd

from typing import Callable, Union
from functools import wraps


def performance_conversion(*arg_names):
    """
    Decorator to convert specific function arguments to numpy arrays and convert
    the return value back to lists.

    This decorator modifies the specified arguments in the decorated function
    by converting them to numpy arrays. It also converts any numpy arrays returned
    by the function back to lists.

    Args:
        *arg_names (str): The names of the arguments to be converted to numpy arrays.

    Returns:
        function: A wrapped function that performs the necessary conversions
        on arguments and return values.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def convert_to_array(value):
                if isinstance(value, list):
                    try:
                        return np.array(value)
                    except ValueError:
                        return [
                            np.array(v) if isinstance(v, list) else v for v in value
                        ]
                return value

            new_kwargs = {
                k: convert_to_array(v) if k in arg_names else v
                for k, v in kwargs.items()
            }

            args = tuple(
                convert_to_array(arg) if arg_name in arg_names else arg
                for arg, arg_name in zip(args, func.__code__.co_varnames)
            )

            result = func(*args, **new_kwargs)

            def convert_result(r):
                if isinstance(r, np.ndarray):
                    return r.tolist()
                elif isinstance(r, tuple):
                    return tuple(convert_result(x) for x in r)
                elif isinstance(r, list):
                    return [convert_result(x) for x in r]
                return r

            return convert_result(result)

        return wrapper

    return decorator


def ensure_numpy_array(func: Callable) -> Callable:
    """
    Decorator to ensure that input data is a numpy array before calling the function.

    This decorator checks if the input data x is a Pandas DataFrame and converts it to
    a numpy array. If x is already a numpy array, it remains unchanged. The function is
    then called with the modified data.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function that ensures the input is a numpy array.
    """

    @wraps(func)
    def wrapper(self, x: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
        x = x.values if isinstance(x, pd.DataFrame) else x
        return func(self, x, *args, **kwargs)

    return wrapper
