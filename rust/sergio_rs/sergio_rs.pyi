from typing import SupportsFloat

import polars

class GRN:
    def __init__(self) -> None: ...
    def add_interaction(
        self, reg: Gene, tar: Gene, k: float, h: float | None, n: float
    ) -> None: ...
    def set_mrs(self) -> None: ...

class Gene:
    def __init__(self, name: str, decay: float) -> None: ...

class Sim:
    def __init__(
        self,
        grn: GRN,
        num_cells: int,
        safety_iter: int,
        scale_iter: int,
        dt: float,
        noise_s: int,
    ) -> None: ...
    def simulate(self, mr_profile: MrProfile) -> polars.DataFrame: ...

class MrProfile:
    @classmethod
    def from_random(
        cls,
        grn: GRN,
        num_cell_types: int,
        low_range: tuple[SupportsFloat, SupportsFloat],
        high_range: tuple[SupportsFloat, SupportsFloat],
    ) -> MrProfile: ...
