import numpy as np
import pandas as pd

from typing import Callable, Union
from functools import wraps


def performance_conversion(*arg_names: str) -> Callable:
    """Creates a decorator to convert specified arguments and return values.

    This decorator factory produces a decorator that, when applied to a
    function, automatically converts specified input arguments from Python
    lists to ``numpy.ndarray`` objects before the function call. It also
    converts ``numpy.ndarray`` objects found in the function's return
    value (including those nested within lists or tuples) back into
    Python lists.

    Argument conversion applies to both positional and keyword arguments
    identified by `arg_names`. If a list cannot be directly converted to a
    ``numpy.ndarray`` due to heterogeneous data (raising a ``ValueError``),
    it attempts to convert nested lists within the main list individually.

    Args:
        *arg_names (str): One or more names of the arguments in the
            decorated function that should be converted from lists to
            ``numpy.ndarray``.

    Returns:
        Callable: The actual decorator that can be applied to a function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Internal helper to convert values to numpy arrays
            def convert_to_array(value):
                if isinstance(value, list):
                    try:
                        return np.array(value)
                    except ValueError:
                        # Attempt to convert nested lists if direct conversion fails
                        return [
                            np.array(v) if isinstance(v, list) else v for v in value
                        ]
                return value

            # Convert specified keyword arguments
            new_kwargs = {
                k: convert_to_array(v) if k in arg_names else v
                for k, v in kwargs.items()
            }

            # Convert specified positional arguments
            # Requires introspection to match arg_names with positional args
            func_arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
            new_args = []
            for i, arg_val in enumerate(args):
                if i < len(func_arg_names) and func_arg_names[i] in arg_names:
                    new_args.append(convert_to_array(arg_val))
                else:
                    new_args.append(arg_val)
            args = tuple(new_args)

            result = func(*args, **new_kwargs)

            # Internal helper to convert numpy arrays in results back to lists
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
    """Decorator to ensure a specific input argument is a ``numpy.ndarray``.

    This decorator is designed for methods where the first argument after
    `self` (conventionally named `x`) is expected to be a ``numpy.ndarray``.
    If this argument is a ``pandas.DataFrame``, it is converted to a
    ``numpy.ndarray`` using its ``.values`` attribute. If it's already a
    ``numpy.ndarray``, it is passed through unchanged.

    Args:
        func (Callable): The method to be decorated. It is assumed to have
            `self` as its first parameter, followed by the data argument `x`.

    Returns:
        Callable: The wrapped method, which will receive `x` as a
            ``numpy.ndarray``.
    """

    @wraps(func)
    def wrapper(self, x: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
        # Convert pandas.DataFrame to numpy.ndarray if necessary
        if isinstance(x, pd.DataFrame):
            x_converted = x.values
        else:
            x_converted = x
        return func(self, x_converted, *args, **kwargs)

    return wrapper
