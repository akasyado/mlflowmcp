import streamlit as st


with open("app/pages/document.md", "r", encoding="utf-8") as f:
    md_text = f.read()

st.markdown(md_text)