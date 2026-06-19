import streamlit as st


with open("pages\document.md", "r", encoding="utf-8") as f:
    md_text = f.read()

st.markdown(md_text)