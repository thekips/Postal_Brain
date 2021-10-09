from typing import Tuple
import sys
import os

Point = Tuple[float, float]

class Spec():
    def __init__(self, shape: tuple, num_values: int, dtype=int, name: str=None) -> None:
        self.shape = shape
        self.num_values = num_values
        self.dtype = dtype
        self.name = name

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout