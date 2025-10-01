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
```
from ExplanationEngine import ExplanationEngine

engine = ExplanationEngine(model) # model : torch.nn.module, only linear and relu layers

# Why explanation
why_res = engine.why(x) # x: np.ndarray (network_input_d,)
print("\nFirst few constraints from WHY explanation:")
A = why_res.get("A")
b = why_res.get("b")
if A is not None and b is not None:
    for i in range(min(5, len(A))):
        print(f"Constraint {i+1}: {np.array2string(A[i][:10], precision=2)}... <= {b[i]:.4f}")
    if len(A) > 5:
        print(f"... and {len(A) - 5} more constraints")

# Why not explanation
why_not_res = engine.why_not(x, counterfactual_class, max_visited=60000) # max polytopes to visit if marching necessary
print("\n" + "="*50)
print("WHY NOT EXPLANATION:")
print(engine.format_why_not_explanation(why_not_res))
print("\n" + "="*50)
```

# Understanding explanations:
A "Why" explanation consists simply of the constraints of the polytope which contains x.
This will consist of n inequalities with d variables, where n is the number of neurons in the network and d the dimensionality of
the input space.

A "Why not" explanation can consist of a single inequality, or many, depending ont he case.
In the case where the counterfactual can be found within the same polytope (CASE: SAME POLYTOPE), this will be a single
inequality of d variables.

In the case where a search was necessary to identify a nearby polytope containing the counterfactual class (CASE: ADJACENT REGION), this will consist of multiple inequalities, namely the constraints which distinguish the polytope P containing x from P' which contains at least one point with the counterfactual class.

By default "Why not" explanations can fail if the counterfactual class is not found within the given search budget (in terms of hamming distance for the polytope containing x), in which case the program will halt without providing an inequality. In theory however halting is not mandatory and the search can continue until all 2^n regions are enumerated, in which case a solution is guaranteed (though in exponential time)