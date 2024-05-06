# Predicting whether a binary will merge from initial conditions

The primary target of population synthesis is to understand the population of binaries that will result is a compact binary coalesence that is observable by gravitational-wave detectors.
However, this is a very rare outcome if initial conditions are chosen randomly.
Methods exist to optimize the choice of initial conditions to maximize the number of merging compact binaries.
In this task, you will develop another approach: training a classifier to predict if a set of initial conditions will result in a merger.

There are two reasons a binary won't result in a compact binary merger:
- a compact binary doesn't form. This can happen for a range of reasons, including the initial stars being too small, or the binary being disrupted before compact objects form.
- a compact binary forms but doesn't merge within the age of the Universe (a.k.a. the [Hubble time](https://en.wikipedia.org/wiki/Hubble%27s_law#Hubble_time)).

In this task, you will attempt to build a classifier that only uses information available at the beginning of the simulation to predict whether the binary will result in a merger (has `Merges_Hubble_Time == 1`).

## Problem

Classification analyses are interested in determining which discrete class and set of initial parameters $\theta$ corresponds to.

For the case here $\theta$ are parameters describing the initial state of a stellar binary and the discrete classes are whether the binary will result in a compact object merger within the age of the Universe.

## Data

The data are available at `/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/compas-data.pkl`.
The final evaluation data consists of data removed randomly from the same inputs.
The function `load_classification_data` will load the data.

## Evaluation

Your models will be evaluated on their ability to make a binary classification about whether a specific set of initial conditions will result in a merging compact binary.
Specifically, we will compute the ["balanced accuracy"](https://en.wikipedia.org/wiki/Confusion_matrix).
This is implemented in the `statistics` function for reference.

## Suggested methods

The example script uses a logistic classifier that directly returns binary classifications.
You could use a more complex classifier, e.g., using deep neural networks or decisions trees.
You could also use class probabilities and manually assign the categories based on a threshold for the probability.

## Extensions

While the balanced accuracy will be used for the quantitative evaluation
there are many other metrics to assess the performance of classification algorithms
explore other metrics described in https://en.wikipedia.org/wiki/Confusion_matrix
and see which can be best optimized.
