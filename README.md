# mlflowmcp

An MCP (Model Context Protocol) server that lets AI assistants interact directly with MLflow. Manage experiments, compare models, and control the model registry through natural conversation instead of switching between the MLflow UI, notebooks, and scripts.

## Features

- **List experiments** - view all MLflow experiments and their runs
- **Compare models** - compare metrics and parameters across runs side by side
- **Retrieve confusion matrices** - pull classification results for a given run
- **Get metrics and parameters** - fetch logged metrics and parameters for any run
- **Push to production** - promote a registered model version to production
- **Remove from production** - transition a model version out of production
- **Create model registry entries** - register new models in the MLflow Model Registry

## Requirements

- Python 3.10+
- A running MLflow tracking server
- An MCP-compatible AI client

## Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/mlflowmcp.git
cd mlflowmcp
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure environment variables

Create a `.env` file in the project root:

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Usage

Once connected, you can ask your AI assistant things like:

- "List all my MLflow experiments"
- "Compare the metrics for runs A and B"
- "Show me the confusion matrix for run XYZ"
- "What parameters were used in the latest run?"
- "Push model with associated run id to production"
- "Remove the current production model"

The assistant translates these requests into MLflow API calls and returns the results conversationally.

## Project Structure

```
mlflowwithmcp/
├── app/
│   ├── MLFLOW_Assistant.py
│   └── pages/
│       ├── 01_Production_Model.py
│       ├── 02_Visualize_Embeddings.py
│       ├── 03_About.py
        └── document.md
├── src/
│   ├── llm_call.py       # (MCP Client and LLM)
│   └── main.py           # (Fastmcp server)
│
├── data/
│   ├── data.csv
│   ├── data_test.csv            # (csv file with 384 vectors of imdb text with text and label)
│   └── embeds_df.csv            # (pca transformed 3d vectors)
```

## Available Tools

| Tool | Description |
|------|-------------|
| `list_experiments` | Returns all experiments tracked by MLflow |
| `runs_in_experiment` | Returns run in experiment |
| `models_in_production` | Retrieves latest model in registry production |
| `push_model_to_production` | Push model a model to registry production |
| `remove_model_from_production` | Removes model from production |
| `get_the_artifacts_path` | Retrieves all artifacts |