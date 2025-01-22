**Overview**

This repository demonstrates a workflow for:

1. Tokenization & [CLS] Embedding Extraction

Using a transformer model (e.g., MacBERT) to tokenize natural language sentences and extract the [CLS] token embedding (which often serves as a representative embedding for the entire sentence).

2. Dimensionality Reduction with t-SNE

Reducing the dimensionality of the high-dimensional embeddings (e.g., 1,024 dimensions from MacBERT) to 2D or 3D space using t-SNE (t-Distributed Stochastic Neighbor Embedding).
t-SNE helps visualize semantic clusters and relationships among sentences in a low-dimensional space.

3. Visualization and Analysis

Using Plotly to interactively visualize the 2D embeddings produced by t-SNE.
Inspecting whether sentences belonging to the same label/class form distinct clusters.
Applying SHAP for interpretability (optional step) to see how the model arrived at a particular prediction.
The example focuses on Chinese sentences containing the verb/particle “起来”, classifying them into five categories (Directional, Resultative, Completive, Inchoative, Discourse).

**Dataset Description**
Training Data: qilai_train_data.txt
Test Data: qilai_test_data.txt
