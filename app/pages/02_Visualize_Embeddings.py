import streamlit as st
import plotly.express as px
import pandas as pd




st.title("Vector Embedding")


    




df = pd.read_csv("data/embeds_df.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


df = df.loc[0:200]



fig = px.scatter_3d(
    x=df["0"],
    y=df["1"],
    z=df["2"],
    title="Embeddings",
    # text = wrapped_comments,
    hover_name=df["wrapped_text"],
    
    color=df["sentiment"],
    color_discrete_map={
        "Positive" : "green",
        "Negative" : "red"
    }
)
# fig.update_traces(
#     textfont=dict(
#         color="white",
#         size=10
#     )
# )
fig.update_layout(
    coloraxis_showscale=False,
    width=1600,
    height=1000
)



st.plotly_chart(fig, use_container_width=True)


