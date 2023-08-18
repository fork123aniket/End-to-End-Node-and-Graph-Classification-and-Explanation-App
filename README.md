# End-to-End Node and Graph Classification and Explanation App

This repo contains project code for ***Graph Explainability*** system that when receives Node ID or Graph ID as user input:
- Classifies the ***node*** or ***graph***;
- Displays ***feature importances*** based on computed ***explained feature mask***;
- Shows ***Explanation Subgraph*** based on ***learned edge mask***;
- Visualizes the original ***node (and its neighborhood)*** and ***graph*** based on which operation does the user wanna perform.

## Requirements
- `Python`
- `Streamlit` (for app building and deployment)
- `DVC` (for data, model, and code version control)
- `MLflow` (to keep track of all graph representation learning experiments performed)
- `Optuna` (to find optimal values for all hyperparameters in exponentially large search space)
- `PyTorch`
- `PyTorch Geometric`

# Data
