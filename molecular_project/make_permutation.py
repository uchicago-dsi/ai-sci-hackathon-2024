import networkx as nx
import json
import numpy as np
import copy
import hashlib

from helper import load_data_from_file, write_data_to_json_file


def remove_param(graph):
    for i in range(len(graph)):
        del graph.nodes[i]["param"]
    return graph

def get_permutation(n, rng):
    permutation = np.asarray(range(n), dtype=int)
    rng.shuffle(permutation)
    return permutation

def get_inv_permutation(permutation):
    inv_permutation = np.argsort(permutation)
    return inv_permutation

def apply_permutation(graph, permutation):
    new_graph = nx.Graph()
    for node in graph.nodes(data=True):
        new_idx = int(permutation[node[0]])
        attrs = node[1]
        new_graph.add_node(new_idx, **attrs)

    for edge in graph.edges(data=True):
        a = int(permutation[edge[0]])
        b = int(permutation[edge[1]])
        attrs = edge[2]
        new_graph.add_edge(a, b, **attrs)
    return new_graph


def write_out_data(data_in, outname):
    data_out = {}
    for smi in data_in:
        m = hashlib.shake_256()
        m.update(bytes(smi, "utf-8"))
        name = m.hexdigest(10)
        graph = remove_param(copy.deepcopy(data_in[smi]))
        data_out[name] = graph
    write_data_to_json_file(data_out, outname, indent=2)

def write_perm_data(data_in, seed, smi, num_permuted_graphs, file_name):
    perm_graphs = {}
    graph = data_in[smi]
    half_graph = remove_param(copy.deepcopy(graph))
    rng = np.random.default_rng(seed=seed)
    permutation_dict = {}
    for _ in range(50):
        perm = get_permutation(len(half_graph), rng)
        m = hashlib.shake_256()
        m.update(perm.tobytes())
        name = m.hexdigest(10)
        permutation_dict[name] = [int(e) for e in perm]

        permuted_graph = apply_permutation(half_graph, perm)
        perm_graphs[name] = permuted_graph
    with open(f"{file_name}.json", "w") as json_handle:
        json.dump(permutation_dict, json_handle)
    write_data_to_json_file(perm_graphs, f"{file_name}_masked.json", indent=2)

def main():
    from util import SEED, SMI
    data_in = load_data_from_file("competition.json")
    write_out_data(data_in, "validation_masked.json")
    write_perm_data(data_in, SEED, SMI, 50, "permutation")

    data_in_tmp = load_data_from_file("data.json")
    # To save space we only us a couple graphs as example
    data_in = {}
    example_num = 10
    for i, smi in enumerate(data_in_tmp):
        data_in[smi] = data_in_tmp[smi]
        if i> example_num:
            break

    write_out_data(data_in, "validation_example.json")
    write_perm_data(data_in, SEED, smi, 5, "permutation_example")
    print(smi)




if __name__ == "__main__":
    main()
