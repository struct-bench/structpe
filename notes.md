#Abstract

Differentially private (DP) synthetic data generation techniques hold great promise to make use of private datasets, which otherwise can't be exposed for model training.
However, the real world utility of such synthetic datasets depends on whether they can adequately mimic the various properties of the real datasets.
Such properties include both form and content and remain challenging to measure.
In this work, we propose Struct-Bench, a benchmark comprising of both real world and synthetic annotated datasets alongside a novel evaluation protocol to characterize the fine-grained properties of DP synthetic datasets in relation to private datasets.
Struct-Bench comprises of 5 real-world and 2 synthetically generated datasets, each annotated with Context-Free Grammars (CFGs), which present a challenging
benchmark for state-of-the-art DP synthetic data generation methods. Struct-Bench also includes reference implementations of different metrics, allowing standarization of different methods.

-- Cite: Model collapse works, since bad synthetic data can be useless. 
