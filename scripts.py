import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import subprocess
import os
from huggingface_hub import snapshot_download



print("="*100)
print("="*100)
print("="*100)
print("="*100)
print("running gdown")
subprocess.run(
    "gdown --folder https://drive.google.com/drive/folders/1TUx3VzG3kypxmlHF2A2gHfevtkM8066p?usp=drive_link",
    shell=True,
    check=True
)
print("gdown complete")
print("="*100)
print("="*100)
print("="*100)
print("="*100)

print("="*100)
print("="*100)
print("="*100)
print("="*100)
print("downloading model")

snapshot_download(
        repo_id="BAAI/bge-small-en-v1.5",
        local_dir="./models/bge-small-en-v1.5"
    )


print("model download complete")
print("="*100)
print("="*100)
print("="*100)
print("="*100)

# os.makedirs("data", exist_ok=True)



df = pd.read_csv("data/data.csv")
df_test = pd.read_csv("data/data_test.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)





param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}

X = df.drop(columns = ["label", "text"])
y = df["label"]

X_test = df_test.drop(columns = ["label", "text"])
y_test = df_test["label"]


experiment_name = "Logistic Regression"
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_name)

for c in param_grid["C"]:
    for penalty in param_grid["penalty"]:
        for solver in param_grid["solver"]:
            with mlflow.start_run(run_name=f"LR_{c}_{penalty}_{solver}"):
                mlflow.log_param("C",c)
                mlflow.log_param("penalty", penalty)
                mlflow.log_param("solver", solver)
                model = LogisticRegression(C = c,
                                        penalty=penalty,
                                        solver=solver,
                                        max_iter=1000)
                model.fit(X, y)

                y_pred = model.predict(X_test)

                accuracy=accuracy_score(y_test,y_pred)

                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
                plt.savefig("confusion_matrix.png")
                mlflow.log_artifact("confusion_matrix.png")

            
                mlflow.log_metric("Accuracy", accuracy)
    
                mlflow.sklearn.log_model(model,name="model", input_example=X[:1])

                print(f"Parameters {c} {penalty} {solver}")




experiment_name = "K neighbours"
mlflow.set_experiment(experiment_name)

for i in range(1,15):
            with mlflow.start_run(run_name=f"KNN_{i}"):
                mlflow.log_param("neighbours",i)
                # mlflow.log_param("penalty", penalty)
                # mlflow.log_param("solver", solver)
                model = KNeighborsClassifier(n_neighbors=i)
                model.fit(X, y)

                y_pred = model.predict(X_test)

                accuracy=accuracy_score(y_test,y_pred)

                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
                plt.savefig("confusion_matrix.png")
                mlflow.log_artifact("confusion_matrix.png")

                
                mlflow.log_metric("Accuracy", accuracy)
    
                mlflow.sklearn.log_model(model,name="model", input_example=X[:1])


                print(f"Parameters {i}")

with open("initialized.txt", "w") as f:
    f.write("True")
