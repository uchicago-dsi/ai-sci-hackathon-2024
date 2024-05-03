# Predicting initial conditions from black hole masses

While having a rapid emulator for the population synthesis simulation, extracting astrophysical information from real observations involves solving the inverse problem: what inputs to the simulation could have led to the observed final state.

In this final challenge you will build a probabilistic emulator for the possible stellar binaries that could have resulted in the given merging binaries.

## Problem

In this inverse problem we want to know the inverse of the mapping learned in the regression task
$$\theta = f^{-1}(\phi).$$
However, there is no reason to expect that this inverse should be unique, especially since the final dimensionality is much less than the input dimensionality.
We are therefore interested in how probably it is that a possible initial values led to the observed binary.
Formally, this can be written using the language of [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference) as
$$p(\theta | \phi)$$.

The most common method of constructing this density using machine learning methods is using [simulation-based inference](https://www.pnas.org/doi/10.1073/pnas.1912789117) (a.k.a., [approximate Bayesian computation](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation).
One method of achieving this is to learn the joint probability density of the initial and end states $p(\theta, \phi)$ in such a way that we can evaluate conditional slices of the density.

## Data

The data are available from `/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/`.
In this directory there are many subdirectory containining files that follow the pattern `Z_{...}/COMPAS_Output.h5`.
The `...` represents the metallicity of the stars considered with all being a random sampling of metallicities.
The final evaluation data consists of data removed randomly from these files and so for best results you should train on all the data.
You can use the same data loading function as for the regression problem.

## Evaluation

The aim is to maximize the probability $p(\theta | \phi)$ for the validation set, the most conventient method to define the loss for this problem is to minimize the negative log-probability.

## Suggested methods

A common choice for analyses such as this are [normalizing flows](http://arxiv.org/abs/1908.09257) or [conditional normalizing flows](https://arxiv.org/abs/1912.00042).
There may be more recent methods available in the literature, or you may be able to get good performance using a simpler method, e.g., Gaussian-mixture models or kernel density estimates.

## Extensions

Which of the initial parameters are easiest to determine from the final parameters?
Are there any parameters that are impossible to constrain?