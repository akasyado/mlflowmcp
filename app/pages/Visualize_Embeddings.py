# from langchain_huggingface import HuggingFaceEmbeddings
# import os
# from huggingface_hub import snapshot_download
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
import streamlit.components.v1 as components
import textwrap

# if not os.path.exists("./models/bge-small-en-v1.5"):
#     snapshot_download(
#         repo_id="BAAI/bge-small-en-v1.5",
#         local_dir="./models/bge-small-en-v1.5"
#     )

# BATCH_SIZE  = 64
# embedding = HuggingFaceEmbeddings(
#     model_name="./models/bge-small-en-v1.5",
#     # model_kwargs={"device": "cuda"},
#     encode_kwargs={
#         "normalize_embeddings": True,
#         "batch_size": BATCH_SIZE
#     }
# )

st.title("Vector Embedding")


    

# label = ["positive"]*99 +["negative"]*99

# pca = PCA(n_components = 3)

# df = pd.read_csv("data/data (1).csv")

# pca = PCA()

# new_df = pca.fit_transform(df.drop(columns=["label", "text"]))

# new_df = pd.DataFrame(new_df)

# new_df["text"] = df["text"]
# new_df["label"] = df["label"]

# new_df.to_csv("data/embeds_df.csv", index = False)

# df = pd.read_csv("data/data.csv")
# df_test = pd.read_csv("data/data_test.csv")

# # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

# pca = PCA(n_components=3)

# array = pca.fit_transform(df_test.drop(columns=["label", "text"]))

# df_diff = pd.DataFrame(array)
# # df["label"] = df["label"].astype(str)

# df_diff["wrapped_text"] = df_test["text"].apply(
#     lambda x: "<br>".join(textwrap.wrap(str(x), width=50))
# )

# df_diff["sentiment"] = df_test["label"].map({0: "Negative", 1: "Positive"})

# df_diff.to_csv("data/embeds_df.csv", index = False)

df = pd.read_csv("data/embeds_df.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# # df["label"] = df["label"].astype(str)

# df["wrapped_text"] = df["text"].apply(
#     lambda x: "<br>".join(textwrap.wrap(str(x), width=50))
# )

# df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})

df = df.loc[0:200]



# pca.fit(df.drop(columns="label"))

# embed = embedding.embed_documents(comments)

# embed = pca.transform(embed)


# embeds_df = pd.DataFrame(embed, columns= [0,1,2])

# embeds_df.to_csv("data/embeds_df.csv", index = False)

# import textwrap

# wrapped_comments = [
#     "<br>".join(textwrap.wrap(c, width=20))
#     for c in comments
# ]

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

# fig.write_html("plot.html")

# with open("plot.html", "r", encoding="utf-8") as f:
#     html = f.read()


# html = fig.to_html(
#     full_html=False,
#     include_plotlyjs="cdn"
# )


# components.html(html, height=900)

st.plotly_chart(fig, use_container_width=True)


