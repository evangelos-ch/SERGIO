import numpy as np
import numpy.typing as npt
import pandas as pd

from SERGIO.GRN import GRN


class sergio:
    def __init__(self, grn: GRN, diff_graph=None) -> None:
        """
        TODO: add functionality for dynamics simulations
        ,
        """

        self.grn_ = grn
        self.diff_graph_ = diff_graph
        self._init_conc()
        self.gNames_ = self._get_gene_names(grn)
        self.lambda_ = self._get_lambda(self.gNames_, grn)

    def _init_conc(self):
        cTypes = list(range(self.grn_.nCellTypes_))
        for g in self.grn_.attr_["genes"].values():
            g.append_sim_conc(conc=g.ss_conc_, cTypes=cTypes)

    def _get_gene_names(self, grn):
        return [k for k in grn.attr_["genes"].keys()]

    def _get_lambda(self, gnames, grn):
        ret = [grn.attr_["genes"][gn].decay_ for gn in gnames]
        return np.array(ret).reshape(-1, 1)

    def _iter_ss(self, noise_ss, dt, cTypes):
        X = np.empty(shape=(len(self.gNames_), len(cTypes)))
        P = np.empty(shape=(len(self.gNames_), len(cTypes)))
        L = self.lambda_
        for ri, gn in enumerate(self.gNames_):
            gene = self.grn_.attr_["genes"][gn]
            X[ri] = gene.get_last_conc(cTypes)
            P[ri] = gene._calc_prod(cTypes, regs_conc="sim")

        P[P < 0] = 0  # numerical stability
        D = L * X
        rndP = np.random.normal(size=(len(self.gNames_), len(cTypes)))
        rndD = np.random.normal(size=(len(self.gNames_), len(cTypes)))

        newX = (
            X
            + (P - D) * dt
            + (np.multiply(np.sqrt(P), rndP) + np.multiply(np.sqrt(D), rndD))
            * noise_ss
            * np.sqrt(dt)
        )
        for gn, conc in zip(self.gNames_, newX):
            self.grn_.attr_["genes"][gn].append_sim_conc(conc.flatten(), cTypes)

    def _simulate_ss(
        self,
        nCells: npt.NDArray,
        noise_ss: float,
        dt: float = 0.01,
        safety_iter: int = 50,
        scale_iter: int = 10,
    ) -> None:
        """
        # TODO: make sure nCells is already np.array
        """
        # first do safety iterations
        cTypes = list(range(self.grn_.nCellTypes_))
        for _ in range(safety_iter):
            self._iter_ss(noise_ss, dt, cTypes)

        # next simulate required iterations
        req = nCells * scale_iter
        cTypes = list(range(self.grn_.nCellTypes_))
        nIter = 0
        while cTypes:
            self._iter_ss(noise_ss, dt, cTypes)
            nIter += 1
            cTypes = np.where(req > nIter)[0].tolist()

    def simulate(
        self,
        nCells: int,
        noise_s: float,
        noise_u=None,
        safety_iter=50,
        scale_iter=10,
        dt=0.01,
    ) -> None:
        if isinstance(nCells, int) or isinstance(nCells, float):
            self.nCells_ = np.array([nCells] * self.grn_.nCellTypes_)
        else:
            assert len(nCells) == self.grn_.nCellTypes_
            self.nCells_ = np.array(nCells)
        self.safety_iter_ = safety_iter
        self.scale_iter_ = scale_iter

        if not self.diff_graph_:
            self._simulate_ss(
                self.nCells_,
                noise_ss=noise_s,
                dt=dt,
                safety_iter=safety_iter,
                scale_iter=scale_iter,
            )
        else:
            raise NotImplementedError

    def _get_rnd_scInd(self):
        ret = {}
        for ct in range(self.grn_.nCellTypes_):
            ret[ct] = self.safety_iter_ + np.random.randint(
                low=0, high=self.nCells_[ct] * self.scale_iter_, size=self.nCells_[ct]
            )
        return ret

    def getSimExpr(self) -> pd.DataFrame:
        rndInd = self._get_rnd_scInd()
        expr = []
        cell_names = []
        gene_names = []
        for g in self.grn_.attr_["genes"].values():
            curr = []
            gene_names.append(g.name_)
            for ct in rndInd.keys():
                ind = rndInd[ct].tolist()
                curr += g.sim_conc_[ct][ind].tolist()
            expr.append(curr)

        cell_names += [
            "type_{}_cell_{}".format(ct, i)
            for ct, n in enumerate(self.nCells_)
            for i in range(n)
        ]
        expr_df = pd.DataFrame(expr)
        expr_df.columns = cell_names  # type: ignore
        expr_df.index = gene_names  # type: ignore

        return expr_df
