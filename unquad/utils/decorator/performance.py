from typing import Callable, Union

import numpy as np
from functools import wraps

import pandas as pd


def performance_conversion(*arg_names):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Helper function to convert lists to numpy arrays, including nested lists
            def convert_to_array(value):
                if isinstance(value, list):
                    try:
                        return np.array(value)
                    except ValueError:
                        # Handle irregular nested lists if necessary
                        return [
                            np.array(v) if isinstance(v, list) else v for v in value
                        ]
                return value

            # Convert kwargs
            new_kwargs = {
                k: convert_to_array(v) if k in arg_names else v
                for k, v in kwargs.items()
            }

            # Convert args
            args = tuple(
                convert_to_array(arg) if arg_name in arg_names else arg
                for arg, arg_name in zip(args, func.__code__.co_varnames)
            )

            # Execute the function with modified arguments
            result = func(*args, **new_kwargs)

            # Convert numpy arrays back to lists in the result
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

    @wraps(func)
    def wrapper(self, x: Union[pd.DataFrame, np.ndarray], *args, **kwargs):
        x = x.values if isinstance(x, pd.DataFrame) else x
        return func(self, x, *args, **kwargs)

    return wrapper
