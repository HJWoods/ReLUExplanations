# ReLUExplanations
Implementation of the algorithms described in "Causal Explanations from the Geometric Properties of ReLU Neural Networks"

# QuickStart:
```
git clone https://github.com/HJWoods/ReLUExplanations
cd ReLUExplanations

# Optionally, create and source a virtual environment first
pip install -r requirements.txt
python mnist.py
```

# Using your own models


# Understanding explanations:
A "Why" explanation consists simply of the constraints of the polytope which contains x.
This will consist of n inequalities with d variables, where n is the number of neurons in the network and d the dimensionality of
the input space.

A "Why not" explanation can consist of a single inequality, or many, depending ont he case.
In the case where the counterfactual can be found within the same polytope (CASE: SAME POLYTOPE), this will be a single
inequality of d variables.

In the case where a search was necessary to identify a nearby polytope containing the counterfactual class (CASE: ADJACENT REGION), this will consist of multiple inequalities, namely the constraints which distinguish the polytope P containing x from P' which contains at least one point with the counterfactual class.

By default "Why not" explanations can fail if the counterfactual class is not found within the given search budget (in terms of hamming distance for the polytope containing x), in which case the program will halt without providing an inequality. In theory however halting is not mandatory and the search can continue until all 2^n regions are enumerated, in which case a solution is guaranteed (though in exponential time)