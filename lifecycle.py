"""Guarantee cleanup even when validation, tokenization or decoding fails."""
from functools import wraps

def clean_generation(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            try:
                return function(self, *args, **kwargs)
            finally:
                if self.handle:
                    self.clear_steering()
                    self.reset()
    return wrapped
