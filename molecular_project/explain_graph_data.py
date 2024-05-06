import networkx as nx
import helper

train_data = helper.load_data_from_file("data.json")


print(f"We have {len(train_data.keys())} molecular graphs to train with.")
print("Please adhere to best practices during training.\n")

# Process the first data point for demonstration purposes
for smiles_string in train_data:
    print(f"Our first data point is described by this SMILES string: {smiles_string}")
    print("For more details on SMILES, visit: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system\n")
    print("Do not rely on the SMILES string for trainig, unless you can build it from the graph!")

    graph = train_data[smiles_string]
    print("The training data consists of networkx graphs:", graph)
    print("All graphs are guaranteed to have fewer than 100 nodes, but there is no strict limit on the number of edges.\n")

    print("Let's examine the organization of the nodes:")
    for node in graph.nodes(data=True):

        print(f"The first index is just a label for the node: {node[0]}.")
        print("Remember, the index has no inherent meaning;")
        print("your results should be consistent even if these indices are permuted.")
        print("Permutation Invariance!\n")

        print("Below are attributes available for node featurization:")
        for attr in ['atomic', 'valence', 'formal_charge', 'aromatic', 'hybridization', "radical_electrons"]:
            print(attr, node[1][attr])

        print("\n'atomic' represents the atomic number, or the number of protons in an atom's nucleus.")
        print("This number differentiates elements, e.g., 6 for carbon and 1 for hydrogen.\n")

        print("Attributes in 'param' are used for supervised learning outputs:")
        for key in ['mass', 'charge', 'sigma', 'epsilon']:
            print(key, node[1]["param"][key])

        print("\nBond type values are categorical, not scalar.")
        print('bond_type_name:', "Refers to the text name found in 'ffbonded.itp' for bond force lookup.", node[1]["param"]['bond_type_name'])
        print('bond_type_id:', "An integer identifier unique to the bond type, usable in training", node[1]["param"]['bond_type_id'])
        print("Initially, you might focus on classifying atoms into these groups, later evolving to predict bond, angle, and dihedral values directly from the graph structure.\n")

        # Limit the output to the first node for clarity
        break

    print("Edges in the graphs are also characterized with attributes usable for model predictions.")
    for edge in graph.edges(data=True):
        print(f"Edge from node {edge[0]} to {edge[1]} is undirected: and has the following properties:")
        for edge_prop in ['type', 'stereo', 'aromatic', 'conjugated']:
            print(edge_prop, edge[2][edge_prop])
        print("For bond specification, refer to: https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html")
        # Limit the output to the first edge for demonstration
        break

    # Stop after processing the first data point to keep the output manageable
    break
