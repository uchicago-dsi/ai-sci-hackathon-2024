#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATADIR = Path("/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/")


def load_regression_data(datadir=DATADIR, metallicity="all"):
    """
    Load the data

    For this we will use publicly available datasets from [zenodo](https://zenodo.org/records/6346444).
    This data is a set of hdf5 files containing information about the initial state, evolution, and end state of the binaries.

    We remove some keys that aren't relevant for this task.
    These are mostly a mixture of discrete flags and entries that are repeated.

    To make the fit easier we downselect the data to only include final masses between 3 and 110.
    """
    ignore = ["Merges_Hubble_Time"]
    
    double_compact_objects = pd.read_pickle(DATADIR / "compas-data.pkl")

    double_compact_objects = double_compact_objects[
        (double_compact_objects["Mass(1)"] < 110)
        & (double_compact_objects["Mass(2)"] < 110)
        & (double_compact_objects["Mass(1)"] > 3)
        & (double_compact_objects["Mass(2)"] > 3)
    ]
    return double_compact_objects


def split_data(data):
    """
    Extract the initial state parameters and regression targets

    We need to extract the initial parameters so that we don't train on information we shouldn't have access to.
    We can do this by extracting any parameters with `ZAMS`$^1$ or `Kick`$^2$ in the name.
    We remove two parameters that enumerate the kind of star at the beginning of the simulation.
 
    1. `ZAMS` (zero-age main sequence) marks when the stars form. Any quantities that are defined at this time are inputs to the simulation.
    2. The kick refers to momentum that is lost during supernova explosions. Technically, these occur later in the simulation, but in practice they are chosen randomly based on some phenomenological prescription.

    The regression targets are the masses of the two compact objects

    Parameters
    ==========
    data: pd.DataFrame
        The input data frame containing the initial and final data

    Returns
    =======
    training_data, test_data, training_labels, test_labels: [pd.DataFrame]
        The data partitioned into training test inputs and labels
    """
    initial_parameters = pd.concat(
        [data.filter(like="ZAMS"), data.filter(like="Kick")],
        join="inner",
        axis=1,
    )
    initial_parameters.describe()
    targets = data[["Mass(1)", "Mass(2)"]]

    return train_test_split(initial_parameters, targets)


if __name__ == "__main__":
    double_compact_objects = load_regression_data()
    double_compact_objects.describe()

    # Before jumping in, remember to perform some visual checks on the data to build some intuition.
    
    plt.scatter(double_compact_objects["Mass@ZAMS(1)"], double_compact_objects["Mass(1)"], s=1)
    cbar = plt.colorbar()
    plt.xlabel("$M_{\\star} [M_{\\odot}]$")
    plt.ylabel("$M_{\\rm CO} [M_{\\odot}]$")
    cbar.set_label("Metallicity / 0.3")
    plt.savefig("regression_initial_vs_final_mass_primary.png")
    plt.close()
    
    plt.scatter(double_compact_objects["Mass@ZAMS(2)"], double_compact_objects["Mass(2)"], s=1)
    cbar = plt.colorbar()
    plt.xlabel("$M_{\\star} [M_{\\odot}]$")
    plt.ylabel("$M_{\\rm CO} [M_{\\odot}]$")
    cbar.set_label("Metallicity / 0.3")
    plt.savefig("regression_initial_vs_final_mass_secondary.png")
    plt.close()

    # Split the data into train/test and input/output
    training_data, test_data, training_labels, test_labels = split_data(double_compact_objects)

    # Using linear regression to fit the final masses.
    # You should replace this with something better!
    reg = LinearRegression().fit(training_data, np.log(training_labels))
    print(f"Training loss: {reg.score(training_data, np.log(training_labels)):.4f}")
    print(f"Test loss: {reg.score(test_data, np.log(test_labels)):.4f}")

    plt.scatter(training_labels["Mass(1)"], np.exp(reg.predict(training_data)[:, 0]), s=1, label="Training")
    plt.scatter(test_labels["Mass(1)"], np.exp(reg.predict(test_data)[:, 0]), s=1, label="Test")
    plt.xlabel("Actual mass $[M_{\\odot}]$")
    plt.ylabel("Predicted mass $[M_{\\odot}]$")
    plt.legend(loc="upper left")
    plt.savefig("regression_predictions.png")
    plt.close()
