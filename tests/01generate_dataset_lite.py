import networkx as nx
import random
import os
import pickle

def random_modification(G, num_modifications):
    G_mod = G.copy()
    nodes = list(G_mod.nodes)
    count = 0

    while count < num_modifications:
        action = random.choice(['add', 'remove'])
        u, v = random.sample(nodes, 2)
        if action == 'add' and not G_mod.has_edge(u, v):
            G_mod.add_edge(u, v)
            count += 1
        elif action == 'remove' and G_mod.has_edge(u, v):
            G_mod.remove_edge(u, v)
            count += 1
    return G_mod

def generate_small_graph_dataset(output_dir, N, n_nodes, pro, mod_levels):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(N):
        for p in pro:
            G = nx.gnp_random_graph(n_nodes, p, seed=i)
            nx.set_node_attributes(G, {v: G.degree[v] for v in G.nodes}, name="label")
            original_path = os.path.join(output_dir, f"G_{n_nodes}_{p}_{i}_orig.gpickle")
            with open(original_path, "wb") as f:
                pickle.dump(G, f)
            
            for mods in mod_levels:
                mod = int(mods * n_nodes)
                G_mod = random_modification(G, mod)
                nx.set_node_attributes(G_mod, {v: G_mod.degree[v] for v in G_mod.nodes}, name="label")
                mod_path = os.path.join(output_dir, f"G_{n_nodes}_{p}_{i}_mod_{mod}.gpickle")
                with open(mod_path, "wb") as f:
                    pickle.dump(G_mod, f)
        print("Kreirao graph: ", "G_"+str(n_nodes)+"_"+str(i)+ ".gpickle")

    return output_dir

# Generate the dataset
output_directory = "tests/data/small_graph_dataset"
for n_nodes in [50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
    generate_small_graph_dataset(
        output_directory, 
        N=5, 
        n_nodes=n_nodes, 
        pro=[0.1, 0.2, 0.3, 0.4], 
        mod_levels=[0.1, 0.2, 0.3])
