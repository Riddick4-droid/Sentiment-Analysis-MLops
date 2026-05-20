import time
import functools
from typing import Callable, Any, Optional
from src.utils.logger import setup_logger

TRACING_ENABLED = True  # Set to False to disable tracing
LOG_ARGS = True
LOG_RETURN = False

_trace_logger = setup_logger("trace")

def trace(func: Optional[Callable] = None, * , log_args:bool=None, log_return:bool=None) ->Callable:
    """
    A decorator to trace function execution time, arguments, and return values.

    Args:
        func: The function to be decorated.
        log_args: Whether to log the function arguments (overrides global setting).
        log_return: Whether to log the function return value (overrides global setting)."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not TRACING_ENABLED:
                return f(*args, **kwargs)
            
            #determine local settings
            local_log_args = log_args if log_args is not None else LOG_ARGS
            local_log_return = log_return if log_return is not None else LOG_RETURN

            func_name = f.__module__ + " . " +f.__qualname__
            start_time = time.perf_counter()

            if local_log_args:
                #truncate large arguments to avoid huge logs
                args_repr = _safe_repr(args, max_len=200)
                kwargs_repr = _safe_repr(kwargs, max_len=200)
                _trace_logger.debug(f"Entering {func_name} | args: {args_repr} |kwargs: {kwargs_repr}")

            else:
                _trace_logger.debug(f"Entering {func_name}")
            
            try:
                result = f(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Log exit with duration
                if local_log_return:
                    result_repr = _safe_repr(result, max_len=200)
                    _trace_logger.debug(f"Exiting {func_name} | duration={duration_ms:.2f}ms | return={result_repr}")
                else:
                    _trace_logger.debug(f"Exiting {func_name} | duration={duration_ms:.2f}ms")

                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                _trace_logger.error(f"Exception in {func_name} after {duration_ms:.2f}ms: {str(e)}", exc_info=True)
                raise

        return wrapper

    # Support both @trace and @trace(log_args=False)
    if func is None:
        return decorator
    else:
        return decorator(func)


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    """Safely get a string representation of an object, truncating if too long."""
    try:
        s = repr(obj)
        if len(s) > max_len:
            return s[:max_len] + "... (truncated)"
        return s
    except Exception:
        return "<unprintable>"


# Example usage (commented out)
if __name__ == "__main__":
    @trace(log_args=True, log_return=True)
    def add(a, b):
        return a + b

    result = add(5, 3)
    print(f"Result: {result}")

    @trace
    def failing_function():
        raise ValueError("Something went wrong")

    try:
        failing_function()
    except:
        pass