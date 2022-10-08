"""Contains the commone functions"""
import streamlit as st

from pathlib import Path


def set_up_initial_ui_layout():
    """Setting up the initial header"""
    html_temp = """<div style="background-color:#25383C; padding:5px">
        <h2 style="color:white;text-align:center;">Resume & Job Description Matcher</h2>
        </div><br>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Please provide the resume and the job description and we'll provide matching score.")
    st.write("")


def get_destination_path(destination):
    """Generate Destination Path w.r.t cwd"""
    path = Path()
    curr_path = str(path.cwd())
    slicer = len(curr_path.split("\\")[-1])
    new_path = curr_path[:-slicer]
    new_path = new_path+destination
    return new_path


def save_files(path, uploaded_files):
    """Takes Resumes Uploaded by User and Save them for further Processing"""
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            with open(path+"\\"+uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())


def empty_data_folder(initial_path):
    """Deletes the unnecessary files at the start of the app."""
    complete_path = get_destination_path(initial_path)
    deletion = [f.unlink() for f in Path(complete_path).glob("*") if f.is_file()]
