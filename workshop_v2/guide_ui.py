from pathlib import Path
import streamlit as st
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
def render():
    st.subheader('Workshop documentation')
    files=sorted((ASSET_ROOT/'docs').glob('*.md'))
    if not files:st.warning('Documentation is not installed yet.');return
    path=st.selectbox('Document',files,format_func=lambda p:p.stem.replace('_',' '))
    text=path.read_text(encoding='utf-8')
    st.download_button('Download this guide',text,file_name=path.name,mime='text/markdown')
    st.markdown(text)
