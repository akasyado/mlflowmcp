import streamlit as st
import pandas as pd
# import plotly.express as px
from mlflow import MlflowClient
import mlflow
from langchain_huggingface import HuggingFaceEmbeddings




if "initialized_production" not in st.session_state:
    st.session_state.uri = mlflow.get_tracking_uri()
    st.session_state.embedding = HuggingFaceEmbeddings(
                                model_name=r".\models\bge-small-en-v1.5",
                                # model_kwargs={"device": "cuda"},
                                encode_kwargs={
                                    "normalize_embeddings": True,
                                    "batch_size": 10
                                }
                            )
    st.session_state.client = MlflowClient(tracking_uri=st.session_state.uri)
    st.session_state.p = st.session_state.client.search_registered_models()
    st.session_state.registry_name = st.session_state.p[0].name
    st.session_state.run_id = st.session_state.p[0].latest_versions[0].run_id
    st.session_state.version = st.session_state.p[0].latest_versions[0].version
    st.session_state.run = st.session_state.client.get_run(run_id = st.session_state.run_id)
    st.session_state.run_name = st.session_state.run.info.run_name
    st.session_state.run_metrics = st.session_state.run.data.metrics
    st.session_state.run_parameters = st.session_state.run.data.params
    st.session_state.experiment_id = st.session_state.run.info.experiment_id
    st.session_state.experiment = st.session_state.client.get_experiment(experiment_id=st.session_state.experiment_id)
    st.session_state.experiment_name = st.session_state.experiment.name
    st.session_state.source = st.session_state.p[0].latest_versions[0].source
    st.session_state.model = mlflow.pyfunc.load_model(st.session_state.source)
    st.session_state.initialized_production = True

info = {
    "experiment_id" : st.session_state.experiment_id,
    "experiment_name" : st.session_state.experiment_name,
    "registry_name" : st.session_state.registry_name,
    "run_id" : st.session_state.run_id,
    "run_name" : st.session_state.run_name,
    "version" : st.session_state.version,
    "metrics" : st.session_state.run_metrics,
    "parameters" : st.session_state.run_parameters
}




# st.json(info)
# st.json({"Embedding Model" : "BAAI/bge-small-en-v1.5"})



st.title("Try Production Model")

user_text = st.chat_input("Write a positive or a negative sentence…")


try:
    embeds = st.session_state.embedding.embed_query(user_text)
    error = None
except Exception as e:
    error = "No Input Available"


col1, col2 = st.columns(2)

with col1:
        st.info("Model Information")
        st.json(info)
        st.json({"Embedding Model" : "BAAI/bge-small-en-v1.5"})

with col2:
    if error:
        st.error(error)

    else:
        st.info("Embeddings size 384")
        st.json(embeds)
        df = pd.DataFrame([embeds],columns =[str(i) for i in range(384)])
        pred = st.session_state.model.predict(df)


        # st.info(f"RESULT: {user_text} is {'negative' if pred == 0 else 'positive'} sentence")
        st.info(user_text)
        st.success(f"RESULT :  {'Negative' if pred == 0 else 'Positive'} Sentiment")
        # st.success(f"{user_text} is {'negative' if pred == 0 else 'positive'} sentence")






