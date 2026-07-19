import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def profile(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info(
            "%s executed in %.3f s",
            func.__name__,
            elapsed,
        )

        return result

    return wrapper