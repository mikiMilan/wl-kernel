import networkx as nx
import random
import os
import pickle

def random_modification(G, num_modifications):
    G_mod = G.copy()
    nodes = list(G_mod.nodes)

    for _ in range(num_modifications):
        action = random.choice(['add', 'remove'])
        u, v = random.sample(nodes, 2)

        if action == 'add' and not G_mod.has_edge(u, v):
            G_mod.add_edge(u, v)
        elif action == 'remove' and G_mod.has_edge(u, v):
            G_mod.remove_edge(u, v)
    return G_mod

def generate_dataset(output_dir, N, n_nodes, pro, mod_levels):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(N):
        for p in pro:
            print("Pocinjem!!!")
            G = nx.gnp_random_graph(n_nodes, p, seed=i)
            original_path = os.path.join(output_dir, f"G_{n_nodes}_{p}_orig.gpickle")
            print("Generisao sam graf!!!")
            with open(original_path, "wb") as f:
                pickle.dump(G, f)
            print("Kreirao graph: ", "G_"+str(n_nodes)+"_"+str(p)+ ".gpickle")

            for mods in mod_levels:
                mod = int(mods * n_nodes)
                G_mod = random_modification(G, mod)
                mod_path = os.path.join(output_dir, f"G_{n_nodes}_{p}_mod_{mod}.gpickle")
                with open(mod_path, "wb") as f:
                    pickle.dump(G_mod, f)
            print("Napravio modifikacije!!!")

    return output_dir

# Generate and save dataset
output_directory = "tests/data/graph_dataset"

# for n_nodes in [1000, 5000]:
#     generate_dataset(output_directory, N=1, n_nodes=n_nodes, pro=[0.1, 0.3], mod_levels=[0.05, 0.1])

# for n_nodes in [10000, 50000]:
#     generate_dataset(output_directory, N=1, n_nodes=n_nodes, pro=[0.01, 0.1], mod_levels=[0.01, 0.05])

for n_nodes in [100000, 500000]:
    generate_dataset(output_directory, N=1, n_nodes=n_nodes, pro= [0.0001, 0.001], mod_levels=[0.001, 0.005])

for n_nodes in [1000000, 5000000]:
    generate_dataset(output_directory, N=1, n_nodes=n_nodes, pro= [0.00001], mod_levels=[0.001, 0.005, 0.0001])