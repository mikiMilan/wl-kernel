import os
import pickle
import numpy as np
import pandas as pd
from wl_kernel.kernel import WLkernel
from scipy.stats import entropy


def compute_metrics(G1, G2, size=1024, iters=3, k=3):
    """Izracunaj WL vektore i sve metrike za par grafova"""
    kernel1 = WLkernel(G1, size=size, k=k)
    kernel2 = WLkernel(G2, size=size, k=k)

    vec1 = kernel1.degree_vector(iter=iters)
    vec2 = kernel2.degree_vector(iter=iters)

    return WLkernel.similarity(vec1, vec2, method="cosine")


def process_graphs(input_dir, output_excel):
    """Prođi kroz sve grafove i zapiši rezultate u Excel"""
    results = []

    for file in os.listdir(input_dir):
        if file.endswith("_orig.gpickle"):
            orig_path = os.path.join(input_dir, file)
            with open(orig_path, "rb") as f:
                G_orig = pickle.load(f)

            base_name = file.replace("_orig.gpickle", "")
            print(base_name)

            # Traži sve modifikovane varijante
            for mod_file in os.listdir(input_dir):
                if mod_file.startswith(base_name) and "mod" in mod_file:
                    mod_path = os.path.join(input_dir, mod_file)
                    with open(mod_path, "rb") as f:
                        G_mod = pickle.load(f)

                    metrics = compute_metrics(G_orig, G_mod)
                    results.append({
                        "graph": base_name,
                        "original": file,
                        "modified": mod_file,
                        "cos  metric": metrics
                    })

    # Upis u Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"Rezultati sačuvani u {output_excel}")


if __name__ == "__main__":
    input_directory = "tests/data/small_graph_dataset"
    output_file = "graph_similarity_results.xlsx"
    process_graphs(input_directory, output_file)
