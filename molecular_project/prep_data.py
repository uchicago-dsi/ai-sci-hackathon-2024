import sys
import rdkit
from rdkit import Chem
import dataclasses
import networkx as nx
import numpy as np

import bigsmiles_gen

from util import get_smiles, SEED
import helper

def make_graph_from_smiles(smi):
    big_mol = bigsmiles_gen.Molecule(smi)
    gen_mol = big_mol.generate()
    ffparam, mol = gen_mol.forcefield_types

    graph = nx.Graph()
    for atomnum in ffparam:
        atom = mol.GetAtomWithIdx(atomnum)
        graph.add_node(atomnum, atomic=atom.GetAtomicNum(),
                       valence=atom.GetTotalValence(),
                       formal_charge=atom.GetFormalCharge(),
                       aromatic=atom.GetIsAromatic(),
                       hybridization=int(atom.GetHybridization()),
                       radical_electrons=int(atom.GetNumRadicalElectrons()),
                       param=dataclasses.asdict(ffparam[atomnum]))
    for node in graph.nodes():
        atom = mol.GetAtomWithIdx(node)
        for bond in atom.GetBonds():
            graph.add_edge(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()),
                           type=int(bond.GetBondType()),
                           stereo=int(bond.GetStereo()),
                           aromatic=bond.GetIsAromatic(),
                           conjugated=bond.GetIsConjugated(),
                           )

    return graph

def prepare_sets(data_size:int, competition_size:int, seed:int, max_node_size:int=100):
    all_smi = []
    all_graphs = []
    smi_gen = get_smiles()
    while len(all_smi) < data_size + competition_size:
        smi = next(smi_gen)
        try:
            graph = make_graph_from_smiles(smi)
        except RuntimeError as exc:
            print(len(all_smi)/(data_size + competition_size), exc)
        except ValueError as exc:
            print(len(all_smi)/(data_size + competition_size), exc)
        except bigsmiles_gen.forcefield_helper.FfAssignmentError as exc:
            print(len(all_smi)/(data_size + competition_size), exc)
        else:
            if len(graph) < max_node_size:
                all_smi += [smi]
                all_graphs += [graph]

    rng = np.random.default_rng(seed=seed)
    idx = list(range(len(all_smi)))
    rng.shuffle(idx)

    shuffled_smi = [all_smi[i] for i in idx]
    shuffled_graphs = [all_graphs[i] for i in idx]

    data_smi = shuffled_smi[:data_size]
    data_graphs = shuffled_graphs[:data_size]

    competition_smi = shuffled_smi[data_size:]
    competition_graphs = shuffled_graphs[data_size:]

    return (data_smi, data_graphs), (competition_smi, competition_graphs)

def write_data(all_smi, all_graphs, name):
    dictionary = {}
    for smi, graph in zip(all_smi, all_graphs):
        dictionary[smi] = graph

    helper.write_data_to_json_file(dictionary, f"{name}.json")

def main(argv):
    if len(argv) != 0:
        raise RuntimeError("Specify exactly one SMILES string")

    (data_smi, data_graphs), (competition_smi, competition_graphs) = prepare_sets(3000, 500, SEED)

    write_data(data_smi, data_graphs, "data")
    write_data(competition_smi, competition_graphs, "competition")


if __name__ == "__main__":
    main(sys.argv[1:])
