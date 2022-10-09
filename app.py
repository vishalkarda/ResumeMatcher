"""
app.py to run the entire code.
"""
import os
import fitz
import joblib
import pdfplumber
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from streamlit_option_menu import option_menu

from model.src.data_preprocessing import DataPreprocess
from model.src.data_preprocessing import RESUME_PATH, JOB_DESCRIPTION_PATH
from model.src.eda import get_top_words
from model.src.similarity_model import SimilarityModel, PredictMatchingScore
from model.src.utils import get_destination_path, save_files
from model.src.utils import set_up_initial_ui_layout, empty_data_folder


def resumes_processing_layout():
    """Main Function to run the module"""
    def extract_data(feed):
        data = []
        with pdfplumber.open(feed) as pdf:
            pages = pdf.pages
            for p in pages:
                data.append(p.extract_text())
        resume_pdf = ' '.join(data)
        return resume_pdf

    resumes_text = []
    resumes_ids = []
    uploaded_resumes = st.file_uploader("Upload one or more Resumes",
                                        accept_multiple_files=True,
                                        type=["pdf"])
    if uploaded_resumes is not None:
        st.write("Files Read")
        max_files = 10
        if len(uploaded_resumes) > max_files:
            st.warning(f"Maximum number of files reached. Only the first {max_files} will be processed.")
            uploaded_resumes = uploaded_resumes[:max_files]
        for resume in uploaded_resumes:
            text = extract_data(resume)
            resumes_text.append(text)
            resumes_ids.append(str(resume.name).split(".")[0])

    dfxa_resume = pd.DataFrame(columns=["ResumeID", "ResumeText"])
    dfxa_resume["ResumeID"] = resumes_ids
    dfxa_resume["ResumeText"] = resumes_text
    dfxa_resume = dfxa_resume.drop_duplicates()

    job_desc_text = []
    jd_ids = []
    uploaded_jds = st.file_uploader("Upload one or more Job Description/s",
                                    accept_multiple_files=True,
                                    type=["pdf"])
    if uploaded_jds is not None:
        max_files = 10
        if len(uploaded_jds) > max_files:
            st.warning(f"Maximum number of files reached. Only the first {max_files} will be processed.")
            uploaded_jds = uploaded_jds[:max_files]
        for jd in uploaded_jds:
            text = extract_data(jd)
            job_desc_text.append(text)
            jd_ids.append(str(jd.name).split(".")[0])

    dfxb_jd = pd.DataFrame(columns=["JobDescID", "JobDescription"])
    dfxb_jd["JobDescID"] = jd_ids
    dfxb_jd["JobDescription"] = job_desc_text
    dfxb_jd = dfxb_jd.drop_duplicates()

    if uploaded_resumes != [] and uploaded_jds != []:
        if st.button("Match Resumes"):
            resumes_after_processing_layout(dfxa_resume, dfxb_jd)


def resumes_after_processing_layout(dfxa_resume, dfxb_jd):
    """"""
    results, eda = st.tabs(["RESULTS", "EDA"])

    with results:
        st.write("Please see below results of the above run.")
        st.write("Below you'll find the resume"
                 " and job description file name and their matching percentage.")
        sm_model = SimilarityModel()
        sm_model = sm_model.get_model()

        dataframe_obj = DataPreprocess()
        # model_data = dataframe_obj.run_preprocessing_pipeline(RESUME_PATH, JOB_DESCRIPTION_PATH)
        model_data = dataframe_obj.run_preprocessing_pipeline_direct(dfxa_resume, dfxb_jd)

        pred_obj = PredictMatchingScore(sm_model)
        model_data = pred_obj.generate_results(model_data)
        # print(model_data, "this what we got let's see")
        st.dataframe(data=model_data[["ResumeID", "JobDescID", "MatchPer"]])

        output_path = get_destination_path("\\model\\data\\model_output")
        joblib.dump(model_data, output_path+"\\"+str("model_output"))

    with eda:
        st.write("Here you can see the basic exploratory analysis of uploaded resumes.")
        topn = 10
        uni_tab, bi_tab, tri_tab = st.tabs(["Unigrams", "Bigrams", "Trigrams"])
        unigrams, bigrams, trigrams = get_top_words(topn, model_data, "ResumeText")
        colors_map = ['#9966ff', '#3399ff', '#00ff00', '#ff6600']
        plt.figure(figsize=(5, 4))

        with uni_tab:
            st.write("Most Used Unigrams In The Resumes Can Be Seen Below.")
            sns.barplot(x='Freq', y='Word', color=colors_map[2], data=unigrams)
            plt.title('Top 10 Unigrams without stopwords', size=10)
            st.pyplot(plt)

        with bi_tab:
            st.write("Most Used Bigrams In The Resumes Can Be Seen Below..")
            sns.barplot(x='Freq', y='Word', color=colors_map[2], data=bigrams)
            plt.title('Top 10 Bigrams Without Stopwords', size=10)
            st.pyplot(plt)

        with tri_tab:
            st.write("Most Used Trigrams In The Resumes Can Be Seen Below.")
            sns.barplot(x='Freq', y='Word', color=colors_map[2], data=trigrams)
            plt.title('Top 10 Trigrams Without Stopwords', size=10)
            st.pyplot(plt)


def resume_ranking_ui():
    """"""
    dashes = "|"*10
    title = dashes+" You can Rank Your Resumes Here "+dashes
    # st.header(title)

    html_temp = """<div style="background-color:#25383C; padding:5px">
            <h3 style="color:white;text-align:center;">"""+str(title)+"""</h3>
            </div><br>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    folder_path = get_destination_path("/model/data/model_output")
    st.write(folder_path)
    if len(os.listdir(folder_path)) == 0:
        st.subheader("Please run a experiment first")
    else:
        output_path = get_destination_path("\\model\\data\\model_output")
        dfxa = joblib.load(output_path+"\\model_output")

        jd_values = ["Default"] + list(dfxa["JobDescID"].unique())
        option = st.selectbox('Please select a Job Description to rank resumes :',
                              jd_values,
                              key="jd_selected")

        if st.button("Rank Resumes"):
            if option in jd_values:
                filtered_data = dfxa[dfxa["JobDescID"] == option]
                filtered_data = filtered_data.sort_values(by=['MatchPer'], ascending=False)
                st.write(filtered_data[["ResumeID", "ResumeText", "MatchPer"]])


def main():
    """Main function to run the file"""
    with st.sidebar:
        choose = option_menu("Main Menu", ["ResumeMatcher", "ResumeRanking", "Author"],
                             icons=['file-slides', 'bar-chart-line', 'person lines fill'],
                             menu_icon="list", default_index=0,
                             styles={
                                 "container": {"padding": "5!important", "background-color": "#000000"},
                                 "icon": {"color": "orange", "font-size": "25px"},
                                 "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                              "--hover-color": "#eee"},
                                 "nav-link-selected": {"background-color": "#24A608"},
                             }
                             )

    if choose == "ResumeMatcher":
        set_up_initial_ui_layout()
        resumes_processing_layout()

    elif choose == "ResumeRanking":
        resume_ranking_ui()

    elif choose == "Author":
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">About the Author</p>', unsafe_allow_html=True)

        st.write(
            "Vishal Karda is a Data Science practitioner, "
            "enthusiast."
            "\n\nHe's also a writer. Who loves reading books, novels and writes excerpts and poems."
            "\n\nTo know more about Vishal, please visit him:"
            "\n\nLinkedIn @ https://www.linkedin.com/in/vishal-karda/")


if __name__ == "__main__":
    empty_data_folder("/model/data/job_descriptions")
    empty_data_folder("/model/data/resumes")
    main()
