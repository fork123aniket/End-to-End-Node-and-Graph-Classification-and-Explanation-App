# :rocket: End-to-End Node and Graph Classification and Explanation App

[![Graph Explainability App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://graph-explainability.streamlit.app/)

This repo contains project code for the ***Graph Explainability*** system that when receives Node ID or Graph ID as user input:
- Classifies the ***node*** or ***graph***;
- Displays ***feature importances*** based on computed ***explained feature mask***;
- Shows ***Explanation Subgraph*** based on ***learned edge mask***;
- Visualizes the original ***node (and its neighborhood)*** and ***graph*** based on which operation does the user wanna perform.

## Requirements
- `Python`
- `Streamlit` (for ***app building and deployment***)
- `DVC` (for ***data, model, and code version control***)
- `MLflow` (to keep track of all ***graph representation learning experiments*** performed)
- `Optuna` (to find ***optimal values for all hyperparameters*** in exponentially large search space)
- `PyTorch`
- `PyTorch Geometric`

## Data
- `Node Classification`: [***Cora***](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) dataset is used that contains a ***homogeneous graph*** comprising ***2708 nodes*** and ***1433 node_features*** along with ***7 class labels***.
- `Graph Classification`: [***ENZYMES***](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html) dataset contains ***600 homogeneous graphs*** along with ***3 node_features*** and the task is to classify any of these graphs in ***6 different Enzymes***.

## App Accessibility
To view and access the app, please click [***here***](https://graphexplanability.streamlit.app/) or type in the following web address to your browser:
[**https://graphexplanability.streamlit.app/**](https://graphexplanability.streamlit.app/)

## App Usage
To learn more about how to use this ***Graph Neural Networks-powered deployed app***, please consider watching the following video:

https://github.com/fork123aniket/End-to-End-Node-and-Graph-Classification-and-Explanation-App/assets/92912434/5f584975-3c9f-4dec-b856-c0d038cf9a61
