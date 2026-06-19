from fastmcp import FastMCP
import mlflow
from mlflow.tracking import MlflowClient
import json
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

mcp = FastMCP("MLFLOW")

# uri = os.environ["MLFLOW_TRACKING_URI"]

mlflow.set_tracking_uri("https://dagshub.com/akasyado/mlflow.mlflow")

client = MlflowClient()

@mcp.resource("mlflow://experiments/all")
def list_experiment() -> str:
    """Get the list of all the experiments"""
    experiments = client.search_experiments()    

    return json.dumps([{"id": exp.experiment_id, "name": exp.name} for exp in experiments])

@mcp.tool()
def runs_in_experiment(experiment_id:List[str]):
    """Given the input of list of experiment id this returns the list of runs with parameters, metrics, name and tags for the list ids of experiment
    You can not put experiment names here get the experiment id of the relevant experiment from list_experiments"""
    result = []
    for id in experiment_id:
        experiment = client.get_experiment(id)
        RUNS = {}
        RUNS[experiment.name] = {}
        runs = client.search_runs(experiment_ids=[id])

        for run in runs:
            RUNS[experiment.name][run.info.run_name] = {}
            RUNS[experiment.name][run.info.run_name]["run_id"] = run.info.run_id
            RUNS[experiment.name][run.info.run_name]["run_metrics"] = run.data.metrics
            RUNS[experiment.name][run.info.run_name]["run_paramters"] = run.data.params
            RUNS[experiment.name][run.info.run_name]["run_tags"] = run.data.tags
        
        result.append(RUNS)

    return result


@mcp.tool()
def models_in_production():
    """List the details of all the version of models in poduction right now"""
    run_id = [(mv.run_id, mv.version) for rm in client.search_registered_models() for mv in rm.latest_versions]
    
    models_info =[]
    for (id,version) in run_id:
        r = {"run_id" : id}
        r["version"] = version
        run = client.get_run(id)
        
        experiment_id = run.info.experiment_id
        r["id"] = experiment_id

        experiment_name = client.get_experiment(experiment_id=experiment_id).name
        r["experiment_name"] = experiment_name
        metrics = run.data.metrics
        r["model_metrics"] = metrics
        parameters = run.data.params
        r["parameters"] = parameters
        models_info.append(r)

    return models_info

@mcp.tool()
def push_model_to_production(run_id : List[str]):
    """This push model into production use this tool if you a list of run_ids"""
    for id in run_id:
        client.create_model_version(
            name="production",
            source=f"runs:/{id}/model",
            run_id=id
        )

    return f"pushed {run_id} into production"

@mcp.tool()
def remove_model_from_production(version: str):
    """Removes the model from production given the version"""

    client.delete_model_version(
    name="production",
    version= version
)

    return f"Removed version {version} from production"



@mcp.tool()
def get_the_artifacts_path(run_id : List[str]):
    """It expects a list of run_ids and returns the stored paths of all the 
    artifacts(confusion matrix) in the given list of run_id"""

    if not isinstance(run_id, list):
        run_id = [run_id]
    artifacts_path = []
    for id in run_id:
        artifact_path = client.get_run(id).info.artifact_uri.replace("file:","")
        artifact_path +=  "/confusion_matrix.png"
        artifacts_path.append(artifact_path)

    return artifacts_path
        


if __name__ == "__main__":
    mcp.run()
