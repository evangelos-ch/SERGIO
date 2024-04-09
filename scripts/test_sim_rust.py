import random

# Seed
import time

import igraph as ig  # type: ignore
import numpy as np
import sergio_rs as sergio

np.random.seed(42)
random.seed(42)


def _setup_grn(n_genes: int = 100):

    # GRN Setup
    topology = ig.Graph.Barabasi(n=n_genes, m=3, directed=True)
    sergio_grn = sergio.GRN()

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
                reg = sergio.Gene(name=topology.vs[i]["name"], decay=decay)
                tar = sergio.Gene(name=topology.vs[j]["name"], decay=decay)
                sergio_grn.add_interaction(reg, tar, k=weight, h=None, n=2)

    sergio_grn.set_mrs()

    return sergio_grn


# Init the GRN
start_time = time.time()
grn = _setup_grn(n_genes=100)

# Initialise simulation
sim = sergio.Sim(
    grn=grn, num_cells=500, safety_iter=150, scale_iter=10, dt=0.01, noise_s=1
)
mr_profile = sergio.MrProfile.from_random(
    grn=grn, num_cell_types=1, low_range=(1, 2.5), high_range=(3.5, 5)
)
df = sim.simulate(mr_profile)
end_time = time.time()


# Add noise
values = df.drop("Genes").to_numpy()
noisy_data = sergio.add_technical_noise(values, sergio.NoiseSetting.DS6)

print(df)
print(f"Runtime: {end_time - start_time}s")
