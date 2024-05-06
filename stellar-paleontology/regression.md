# Predicting black hole masses from initial conditions

One of the primary targets of binary population synthesis is the final state of the two objects.
This is described in terms of the final mass of the two objects.
This can then be directly compared to the observed population of binary black hole mergers to test the assumptions of the model.

In COMPAS this involves simulating the full lifetime of the binary from initial stars to final objects.
In this challenge, you will learn a direct mapping from initial parameters to final masses.
We are specifically interested in binaries that result in a compact object binary.

In practice, there is some randomness to the evoution of the binary, for this reason, a probabilistic model may be able to better reproduce the behaviour of the detailed simulations.
In a later task, you will extend this method to use a probabilistic model that can map from either beginning to end state, or end to beginning.

## Problem

Regression analyses attempt to learn the relationships between independent variables $\theta$ and dependent variables $\phi$
$$\phi = f(\theta)$$.
The aim is to generate a deterministic approximation for the mapping $f$.

For the case here $\theta$ are parameters describing the initial state of a stellar binary and $\phi$ are parameters describing the final state of a merging compact binary.

## Data

The data are available in `/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/compas-data.pkl`.
The final evaluation data consists of data removed randomly from the same inputs.
The function `load_regression_data` will load the data.

## Evaluation

Your models will be evaluated using a fraction squared error loss in the predicted vs true final two-dimensional masses.

## Suggested methods

The example script uses ordinary least squares linear regression.
You could use a more complex regressor, e.g., using deep neural networks or decisions trees.
You could also different pre-processing methods of the data to extract features manually to improve the quality of the fit.

## Extensions

There may be other combinations of final parameters which can be more easily predicted, what is the combination of masses that can be most well predicted?
Can you predict other quantites, e.g., `Coalescence_Time` or `Ecceentricity/SemiMajorAxis@DCO`?
