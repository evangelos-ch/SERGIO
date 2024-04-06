import random
import time

import igraph as ig  # type: ignore
import numpy as np

import SERGIO
import SERGIO.GRN
import SERGIO.MR

# Seed
np.random.seed(42)
random.seed(42)


def _setup_grn(n_genes: int = 100, n_types: int = 1):

    # GRN Setup
    topology = ig.Graph.Barabasi(n=n_genes, m=3, directed=True)
    sergio_grn = SERGIO.GRN.GRN()

    # Add node names
    for i, vertex in enumerate(topology.vs):
        vertex["name"] = "GENE" + str(i + 1).zfill(4)

    # Sample weights
    for i, edge in enumerate(topology.es):
        if np.random.random() < 0.5:
            k_range = [-5, -1]
        else:
            k_range = [1, 5]
        edge["weight"] = np.random.uniform(low=k_range[0], high=k_range[1])

    # Create an adjacency matrix with (-1, 0, 1) values
    adjacency_matrix = np.array(topology.get_adjacency())
    for i, row in enumerate(adjacency_matrix):
        for j, is_connected in enumerate(row):
            if is_connected:
                weight = topology.es[topology.get_eid(i, j)]["weight"]
                if weight < 0:
                    adjacency_matrix[i, j] = -1
                # and also add interactions to the GRN
                decay = 0.8
                reg = SERGIO.GRN.Gene(name=topology.vs[i]["name"], decay=decay)
                tar = SERGIO.GRN.Gene(name=topology.vs[j]["name"], decay=decay)
                inter = SERGIO.GRN.SingleInteraction(
                    reg=[reg], tar=tar, k=weight, h=None, n=2
                )
                sergio_grn.add_interaction(inter)

    # Init the GRN
    sergio_grn.setMRs()
    mr_profile = SERGIO.MR.mrProfile(MR_names=sergio_grn.get_mrs(), n_types=n_types)
    mr_profile.build_rnd(range_dict={"L": [1, 2.5], "H": [3.5, 5]})
    sergio_grn.init(mr_profile, update_half_resp=True)

    return sergio_grn


start_time = time.time()
grn = _setup_grn(n_genes=100, n_types=1)

# Initialise simulation
sim = SERGIO.sergio(grn)

# Simulate a steady state
sim.simulate(nCells=500, noise_s=1, safety_iter=150, scale_iter=10)
end_time = time.time()

print(sim.getSimExpr())
print(f"Runtime: {end_time - start_time}s")
