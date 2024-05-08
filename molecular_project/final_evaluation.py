import networkx as nx
import numpy as np
import helper
import hashlib
import copy
import json

from make_permutation import get_inv_permutation, apply_permutation

def compare_property(property_name:str, result_dict, ref_dict, max_node_size:int=100):
    def get_graph_property_data(property_name, graph):
        data = np.zeros(max_node_size)
        for node in graph.nodes(data=True):
            data[node[0]] = node[1]["param"][property_name]
        return data
    result_data = []
    ref_data = []

    miss_counter = 0
    node_counter = 0
    for smi in ref_dict:
        m = hashlib.shake_256()
        m.update(bytes(smi, "utf-8"))
        name = m.hexdigest(10)
        if name not in result_dict:
            miss_counter += 1
        else:
            node_counter += len(ref_dict[smi])
            result_data.append(get_graph_property_data(property_name, result_dict[name]))
            ref_data.append(get_graph_property_data(property_name, ref_dict[smi]))
    result_data = np.asarray(result_data)
    ref_data = np.asarray(ref_data)

    negative_counter = np.sum(ref_data < 0)

    print(f"\nAnalysis for {property_name}")
    print(f"# Missing Molecules: {miss_counter}")
    print(f"# Negative values detected: {negative_counter}")

    sq_diff = (result_data - ref_data)**2
    RMS = np.sqrt(np.sum(sq_diff)/node_counter)
    MAX = np.sqrt(np.max(sq_diff))
    print(f"Root Mean Squared Difference: {RMS}")
    print(f"Max difference: {MAX}")
    print("\n")
    return RMS, MAX, miss_counter, negative_counter


def add_data_from_prediction(result_dict, rng):
    """
    This function simulates adding data to a final graph.
    Do not use this function.
    Use your model to do the prediction and then fill it in.
    """
    def get_random_param(rng):
        param = {}
        param["epsilon"] = rng.random()
        param["mass"] = rng.random()
        param["sigma"] = rng.random()
        param["charge"] = rng.normal()
        return param

    for name in result_dict:
        graph = result_dict[name]
        for node in graph.nodes(data=True):
            node[1]["param"] = get_random_param(rng)
            graph.update(nodes=[node])

    # Remove a graph for good measure
    del result_dict[name]
    # Add a different graph
    result_dict["asdf"] = graph

def compare_permutation(property_name:str, result_dict, ref_graph, permutation_dict, max_node_size:int=100):
    def get_graph_property_data(property_name, graph):
        data = np.zeros(max_node_size)
        for node in graph.nodes(data=True):
            data[node[0]] = node[1]["param"][property_name]
        return data
    result_data = []
    ref_data = []

    miss_counter = 0
    node_counter = 0
    for name in permutation_dict:
        perm = np.asarray(permutation_dict[name], dtype=int)
        try:
            inv_permutation = get_inv_permutation(perm)
            graph = result_dict[name]
            inv_graph = apply_permutation(graph, inv_permutation)
            result_data.append(get_graph_property_data(property_name, inv_graph))
            ref_data.append(get_graph_property_data(property_name, ref_graph))
            node_counter += len(ref_graph)
        except Exception as exc:
            print(exc)
            miss_counter += 1

    result_data = np.asarray(result_data)
    ref_data = np.asarray(ref_data)

    negative_counter = np.sum(ref_data < 0)

    print(f"\nAnalysis for {property_name}")
    print(f"# Missing Molecules: {miss_counter}")
    print(f"# Negative values detected: {negative_counter}")

    sq_diff = (result_data - ref_data)**2
    RMS = np.sqrt(np.sum(sq_diff)/node_counter)
    MAX = np.sqrt(np.max(sq_diff))
    print(f"Root Mean Squared Difference: {RMS}")
    print(f"Max difference: {MAX}")
    print("\n")

    return RMS, MAX, miss_counter, negative_counter



def main():
    # You won't necessarily have this data available, here we use the training data to show you
    ref_dict = helper.load_data_from_file("data.json")


    result_dict = helper.load_data_from_file("validation_example.json")
    # In a real case, we would not add random results.
    # Instead you would fill it with your results and then write it to disk to hand it to us.
    # This is for mock up testing only.
    rng = np.random.default_rng()
    add_data_from_prediction(result_dict, rng)


    compare_property("epsilon", result_dict, ref_dict)
    compare_property("mass", result_dict, ref_dict)
    compare_property("sigma", result_dict, ref_dict)
    compare_property("charge", result_dict, ref_dict)


    print("Permutation check")
    # The real SMILES string we will test with is not in your training data
    ref_graph = ref_dict["O=C(c1ccc2c(c1)OCO2)c1ccc2n1CCC2C(=O)O"]
    result_perm_dict = helper.load_data_from_file("permutation_example_masked.json")
    with open("permutation_example.json", "r") as json_handle:
        permutation_dict = json.load(json_handle)
    # This step is again replaced with your model data
    add_data_from_prediction(result_perm_dict, rng)

    compare_permutation("epsilon", result_perm_dict, ref_graph, permutation_dict)
    compare_permutation("mass", result_perm_dict, ref_graph, permutation_dict)
    compare_permutation("sigma", result_perm_dict, ref_graph, permutation_dict)
    compare_permutation("charge", result_perm_dict, ref_graph, permutation_dict)




if __name__ == "__main__":
    main()
