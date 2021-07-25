from typing import Tuple

Point = Tuple[float, float]

class Spec():
    def __init__(self, shape: tuple, num_values: int, dtype=int, name: str=None) -> None:
        self.shape = shape
        self.num_values = num_values
        self.dtype = dtype
        self.name = name