"""Process the Inputs Provided by User """
import glob
import re
import fitz
import string
import pandas as pd
import pdftotext


from pathlib import Path

HALF_WORDS = {"ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not",
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}
JOB_DESCRIPTION_PATH = "\\model\\data\\job_descriptions"
RESUME_PATH = "\\model\\data\\resumes"


class DataPreprocess:
    def __init__(self):
        self.job_descriptions = []
        self.job_description_ids = []
        self.jds_df = None
        self.resumes = []
        self.resume_ids = []
        self.resumes_df = None

    def set_resumes(self, resumes_path):
        """"""
        file_type = '\*.pdf'
        resumes, ids = self.read_all_the_files(resumes_path, file_type)
        print(resumes, "data after being read")
        dfxa = pd.DataFrame(columns=["ResumeID", "ResumeText"])
        dfxa["ResumeID"] = ids
        dfxa["ResumeText"] = resumes
        self.resumes_df = dfxa

    def get_resumes_df(self):
        """Getter Method for Resumes"""
        return self.resumes_df

    def get_job_descriptions_df(self):
        """Getter Method for Job Descriptions"""
        return self.jds_df

    def set_job_description(self, jd_path):
        """"""
        file_type = '\*.pdf'
        job_descriptions, ids = self.read_all_the_files(jd_path,
                                                        file_type)
        dfxa = pd.DataFrame(columns=["JobDescID", "JobDescription"])
        dfxa["JobDescID"] = ids
        dfxa["JobDescription"] = job_descriptions
        self.jds_df = dfxa

    @staticmethod
    def generate_path(destination):
        """"""
        # path = Path()
        # curr_path = str(path.cwd())
        # slicer = len(curr_path.split("\\")[-1])
        # new_path = curr_path[:-slicer]
        # new_path = new_path + destination

        path = Path()
        curr_path = str(path.cwd())
        new_path = curr_path + destination
        return new_path

    def read_all_the_files(self, destination, file_type):
        """"""
        destination = self.generate_path(destination)
        print("FINAL PATH IS ", destination)
        print("FINAL TWO PATH IS ", str(destination)+file_type)
        filenames = glob.glob(str(destination)+file_type)
        print(filenames)
        files = []
        ids = []

        for item in filenames:
            ids.append(item.split("\\")[-1].split(".")[0])
            with fitz.open(item) as doc:
                temp = []
                for page in doc:
                    text = page.get_text()
                    temp.append(text)
                resume = ' '.join(temp)
                files.append(resume)

        return files, ids

    @staticmethod
    def text_preprocess(df, column_name):
        """"""
        df[column_name] = df[column_name].apply(lambda x: str(x).replace("\n", " "))

        # Regular expression for to complete words
        half_words_re = re.compile('(%s)' % '|'.join(HALF_WORDS.keys()))

        # Function for expanding half words
        def expand_halfwords(text, half_words_dict=HALF_WORDS):
            def change(match):
                return half_words_dict[match.group(0)]

            return half_words_re.sub(change, text)

        df[column_name] = df[column_name].apply(lambda x: expand_halfwords(x))

        df[column_name] = df[column_name].apply(lambda x: x.lower())
        df[column_name] = df[column_name].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
        # Removing extra spaces
        df[column_name] = df[column_name].apply(lambda x: re.sub(' +', ' ', x))

        return df

    def run_preprocessing_pipeline(self, resume_path, jd_path):
        """"""
        self.set_resumes(resume_path)
        self.set_job_description(jd_path)

        print(resume_path, "RESUME PATH")
        print(jd_path, "JD PATH")

        resumes = self.get_resumes_df()
        resumes = self.text_preprocess(resumes, "ResumeText")

        job_desc = self.get_job_descriptions_df()
        job_desc = self.text_preprocess(job_desc, "JobDescription")

        updated_df = resumes.assign(key=1).merge(job_desc.assign(key=1), on='key').drop('key', axis=1)

        print(updated_df)
        return updated_df


if __name__ == "__main__":
    obj = DataPreprocess()
    obj.set_resumes(RESUME_PATH)
    obj.set_job_description(JOB_DESCRIPTION_PATH)

    # resumes_ = obj.resumes_df
    # jds_ = obj.jds_df
    #
    # dfxb = obj.text_preprocess(resumes_, "ResumeText")
    # print(dfxb)
    #
    # dfxb = obj.text_preprocess(jds_, "JobDescription")
    # print(dfxb)
    obj.run_preprocessing_pipeline(RESUME_PATH, JOB_DESCRIPTION_PATH)




