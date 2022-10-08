"""
main_app.py to run the entire code.
"""
import joblib
import matplotlib.pyplot as plt
import os
import seaborn as sns
import streamlit as st

from streamlit_option_menu import option_menu

import model.src.data_preprocessing as data_prep
from model.src.data_preprocessing import DataPreprocess
from model.src.data_preprocessing import RESUME_PATH, JOB_DESCRIPTION_PATH
from model.src.eda import get_top_words
from model.src.similarity_model import SimilarityModel, PredictMatchingScore
from model.src.utils import get_destination_path, save_files
from model.src.utils import set_up_initial_ui_layout, empty_data_folder


def resumes_processing_layout():
    """Main Function to run the module"""
    uploaded_resumes = st.file_uploader("Upload one or more Resumes",
                                        accept_multiple_files=True,
                                        type=["pdf"])

    resume_path = get_destination_path("\\model\\data\\resumes")
    if uploaded_resumes is not None:
        max_files = 10
        if len(uploaded_resumes) > max_files:
            st.warning(f"Maximum number of files reached. Only the first {max_files} will be processed.")
            uploaded_resumes = uploaded_resumes[:max_files]
        save_files(resume_path, uploaded_resumes)

    uploaded_jds = st.file_uploader("Upload one or more Job Description/s",
                                    accept_multiple_files=True,
                                    type=["pdf"])
    jd_path = get_destination_path("\\model\\data\\job_descriptions")
    if uploaded_jds is not None:
        max_files = 10
        if len(uploaded_jds) > max_files:
            st.warning(f"Maximum number of files reached. Only the first {max_files} will be processed.")
            uploaded_jds = uploaded_jds[:max_files]
        save_files(jd_path, uploaded_jds)

    if uploaded_resumes != [] and uploaded_jds != []:
        if st.button("Match Resumes"):
            resumes_after_processing_layout()


def resumes_after_processing_layout():
    """"""
    results, eda = st.tabs(["RESULTS", "EDA"])

    with results:
        st.write("Please see below results of the above run.")
        st.write("Below you'll find the resume"
                 " and job description file name and their matching percentage.")
        sm_model = SimilarityModel()
        sm_model = sm_model.get_model()
        pred_obj = PredictMatchingScore(sm_model)

        dataframe_obj = DataPreprocess()
        model_data = dataframe_obj.run_preprocessing_pipeline(RESUME_PATH, JOB_DESCRIPTION_PATH)

        model_data = pred_obj.generate_results(model_data)
        print(model_data, "this what we got let's see")
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


def checking_something():
    """"""
    dashes = "|"*10
    title = dashes+" You can Rank Your Resumes Here "+dashes
    # st.header(title)

    html_temp = """<div style="background-color:#25383C; padding:5px">
            <h3 style="color:white;text-align:center;">"""+str(title)+"""</h3>
            </div><br>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    folder_path = get_destination_path("\\model\\data\\model_output")
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
        checking_something()

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
    empty_data_folder("\\model\\data\\job_descriptions")
    empty_data_folder("\\model\\data\\resumes")
    main()
