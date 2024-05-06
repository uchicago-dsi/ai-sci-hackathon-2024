#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATADIR = Path("/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/")


def load_classification_data(datadir=DATADIR, metallicity="all"):
    """
    Load the data
    
    We read the data from the prepepared pickle file.
    
    We have retained parameters that have `ZAMS`$^1$ or `Kick`$^2$ in the name along with our classification target.

        1. `ZAMS` (zero-age main sequence) marks when the stars form. Any quantities that are defined at this time are inputs to the simulation.
        2. The kick refers to momentum that is lost during supernova explosions. Technically, these occur later in the simulation, but in practice they are chosen randomly based on some phenomenological prescription.

    We also retain our target, `Merges_Hubble_Time`"
    """
    ignore = ["Mass(1)", "Mass(2)", "Eccentricity@DCO", "SemiMajorAxis@DCO", "Coalescence_Time"]

    double_compact_objects = pd.read_pickle(datadir / "compas-data.pkl")
    for key in ignore:
        if key in double_compact_objects:
            del double_compact_objects[key]
    return double_compact_objects


def statistics(confusion: np.ndarray, verbose=True) -> float:
    """
    Compute summary statistics based on the confusion matrix.

    For more details see https://en.wikipedia.org/wiki/Confusion_matrix.

    As we are interested in both reducing false positives and negatives, quantitative
    evaluation of the model will be based on the balanced accuracy.

    Parameters
    ==========
    confusion: np.ndarray
        The two-by-two confusion matrix of predicted class vs actual class

    Returns
    =======
    balanced_accuracy: float
        The balanced accuracy of the predictions
    """
    confusion = confusion / confusion.sum(axis=0)
    true_positive_rate = confusion[1, 1]
    true_negative_rate = confusion[0, 0]
    false_positive_rate = confusion[1, 0]
    false_negative_rate = confusion[0, 1]
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

    if verbose:
        print(f"True positive rate: {true_positive_rate:.4f}")
        print(f"False positive rate: {false_positive_rate:.4f}")
        print(f"True negative rate: {true_negative_rate:.4f}")
        print(f"False negative rate: {false_negative_rate:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    return balanced_accuracy


def create_and_fit_simple_logistic_classifier(
    inputs: pd.DataFrame, target: pd.Series
) -> (LogisticRegression, StandardScaler):
    """
    Train/run our classifier

    For a simple example, we use logistic regression as implemented in `scikit-learn`.
    This does a terrible job, hopefully you can do better!

    First we scale the input data to have zero mean and unit variance.

    Parameters
    ==========
    inputs: pd.DataFrame
        Data frame of input training data
    target: pd.Series
        Labels corresponding to training set

    Returns
    =======
    reg: LogisticRegression
        The trained regressor that can be used for prediction
    scaler: StandardScaler
        An object for scaling the test inputs for pre-processing
    """
    scaler = StandardScaler().fit(inputs)
    scaled = scaler.transform(inputs)
    
    reg = LogisticRegression().fit(scaled, target)
    print(f"Score on training data: {reg.score(scaled, target):.2f}")
    return reg, scaler


if __name__ == "__main__":
    # Load and inspect the data
    
    double_compact_objects = load_classification_data()
    print(double_compact_objects.describe())
    
    # We can look at how correlated the parameters are, the more strongly correlated something is
    # with having a merger the more likely it is to be a good indicator
    print(double_compact_objects.corr())
    
    
    # As an initial check, we plot the initial masses of the two objects colored by whether they merge within the age of the Universe.
    # 
    # If the purple and yellow dots are clearly disjoint, this will be an easy exercise.
    # They are not.
    
    plt.scatter(
        double_compact_objects["Mass@ZAMS(1)"],
        double_compact_objects["Mass@ZAMS(2)"],
        s=1,
        c=double_compact_objects["Merges_Hubble_Time"],
    )
    cbar = plt.colorbar()
    plt.xlabel("More massive initial stellar mass $[M_{\\odot}]$")
    plt.ylabel("Less massive initial stellar mass $[M_{\\odot}]$")
    cbar.set_label("Merges")
    plt.savefig("masses_vs_merges.png")
    plt.close()
    
    plt.scatter(
        double_compact_objects["SemiMajorAxis@ZAMS"],
        double_compact_objects["Mass@ZAMS(1)"] + double_compact_objects["Mass@ZAMS(2)"],
        c=double_compact_objects["Merges_Hubble_Time"],
        s=1,
    )
    plt.xlabel("Initial semi-major axis [AU]")
    plt.ylabel("Initial total binary mass $[M_{\\odot}]$")
    plt.savefig("semimajoraxis_vs_merges.png")
    plt.close()
    
    # Prepare data for training.
    # Our target is when simply the `Merges_Hubble_Time` field.
    # We convert to an `int` because in the data, this is stored as an unsigned integer which can cause issues.
    # We split the data into a training and test set using a standard sklearn utility.
    
    training_data, test_data = train_test_split(double_compact_objects)
    training_labels = training_data.pop("Merges_Hubble_Time").astype(int)
    test_labels = test_data.pop("Merges_Hubble_Time").astype(int)
    reg, scaler = create_and_fit_simple_logistic_classifier(
        training_data, training_labels
    )
    scaled = scaler.transform(test_data)
    print(f"Score on test data: {reg.score(scaled, test_labels):2f}")
    
    predictions = reg.predict(scaled)
    
    mergers = test_labels == 1
    non_mergers = test_labels == 0
    
    plt.hist(predictions[mergers], bins=100, density=False, histtype="step", label="Merges")
    plt.hist(predictions[non_mergers], bins=100, density=False, histtype="step", label="Does not merge")
    plt.yscale("log")
    plt.xlabel("Predicted $p_{\\rm merges}$")
    plt.ylabel("Probability")
    plt.legend(loc="best")
    plt.xlim(0, 1)
    plt.savefig("classification_probabilities.png")
    plt.close()
    
    confusion = confusion_matrix(test_labels, predictions)
    print(confusion)
    plt.imshow(confusion)
    plt.savefig("classification_confusion_matrix.png")
    plt.close()
    
    score = statistics(confusion)
