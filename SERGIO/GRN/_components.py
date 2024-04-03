from typing import Literal

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ._utils import getInterName


class Gene:
    def __init__(self, name: str, decay: float = 0.8) -> None:
        self.name_ = name
        self.isMR_ = False
        self.level: int | None = None
        self.regs: list[Self] = []
        self.tars: list[Self] = []
        # 'reg names (seperated by comma if coop)' --> interactions
        self.inInteractions: dict[str, SingleInteraction] = {}
        self.prod_rates_: npt.NDArray | None = None
        self.ss_conc_: npt.NDArray | None = None
        self.sim_conc_: npt.NDArray | None = None
        self.current_iters: npt.NDArray | None = None
        self.decay_ = decay
        # self.outInteractions = {} # 'target names (seperated by comma if coop)' --> interactions
        # self.isTF_ = TF

    def _calc_prod(
        self, cTypes: npt.NDArray, regs_conc: Literal["ss", "sim"] = "ss"
    ) -> npt.NDArray:
        if self.isMR_:
            assert self.prod_rates_ is not None
            return self.prod_rates_[cTypes]  # np.arr

        ret = np.zeros(shape=(len(cTypes),))
        for inter in self.inInteractions.values():
            ret += inter._get_hill(cTypes, regs_conc=regs_conc)

        return ret

    def append_sim_conc(self, conc: npt.NDArray, cTypes: npt.NDArray) -> None:
        assert len(conc) == len(cTypes)
        assert self.sim_conc_ is not None and self.current_iters is not None

        if len(cTypes) == len(self.sim_conc_):
            # Special case
            self.sim_conc_[:, self.current_iters[0]] = np.maximum(0.0, conc)
            self.current_iters += 1
        else:
            # prevents negative expression
            self.sim_conc_[cTypes, self.current_iters[cTypes]] = np.maximum(0.0, conc)
            self.current_iters[cTypes] += 1

    def get_last_conc(self, cTypes: npt.NDArray) -> npt.NDArray:
        assert self.sim_conc_ is not None and self.current_iters is not None
        if len(cTypes) == len(self.sim_conc_):
            # Special case
            return self.sim_conc_[:, self.current_iters[0] - 1]
        else:
            return self.sim_conc_[cTypes, self.current_iters[cTypes] - 1]  # type: ignore


# todo make a base interaction object
@numba.jit
def _get_hill_inner(x, n, h, k):
    ret = np.power(x, n) / (np.power(x, n) + np.power(h, n))
    if k > 0:
        return k * ret
    else:
        return np.abs(k) * (1 - ret)


class SingleInteraction:
    def __init__(self, reg: list[Gene], tar: Gene, k=None, h=None, n=None):
        self.reg_: list[Gene] = reg  # can be list if coop edge --> always be list
        self.tar_: Gene = tar
        self.name_ = getInterName(reg, tar)
        self.k_ = k
        self.h_ = h
        self.n_ = n

    def _get_hill(self, cTypes: npt.NDArray, regs_conc="ss") -> npt.NDArray:
        """
        cTypes : is list of cell type ids to compute hill for
        regs_conc (str): 'ss' (uses the steady state concentration of all regs),
        'sim' (uses the last simulated concentration of all regs)
        """
        if len(self.reg_) > 1:
            raise NotImplementedError("Coop interactions are not yet implemented")

        if regs_conc == "ss":
            assert self.reg_[0].ss_conc_ is not None
            x = self.reg_[0].ss_conc_[cTypes]  # np.arr
        else:
            x = self.reg_[0].get_last_conc(cTypes)  # np.arr
        return _get_hill_inner(x, self.n_, self.h_, self.k_)
