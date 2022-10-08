from sentence_transformers import SentenceTransformer, util

from data_preprocessing import DataPreprocess
from data_preprocessing import RESUME_PATH, JOB_DESCRIPTION_PATH


class SimilarityModel:

    @staticmethod
    def __load_model(model_name):
        """"""
        model = SentenceTransformer(model_name)
        return model

    def get_model(self, model_name='all-MiniLM-L6-v2'):
        """"""
        return self.__load_model(model_name)


class PredictMatchingScore:

    def __init__(self, model):
        self.__model = model

    def __generate_embeddings(self, data):
        """"""
        embedding = self.__model.encode(data, convert_to_tensor=True)
        return embedding

    @staticmethod
    def __calculate_similarity(embeddings1, embeddings2):
        """"""
        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        return cosine_scores

    def generate_results(self, dataframe):
        """"""
        resumes = [res.strip() for res in dataframe["ResumeText"].values]
        job_description = [jd.strip() for jd in dataframe["JobDescription"].values]

        embeddings1 = self.__generate_embeddings(resumes)
        embeddings2 = self.__generate_embeddings(job_description)

        cosine_scores = self.__calculate_similarity(embeddings1, embeddings2)

        predicted_ss = []

        # Output the pairs with their score
        for i in range(len(resumes)):
            predicted_ss.append(round(float(cosine_scores[i][i]), 2))

        dataframe["MatchPer"] = predicted_ss

        return dataframe


if __name__ == "__main__":
    sm_model = SimilarityModel()
    sm_model = sm_model.get_model()
    print("Similarity Model Done")
    pred_obj = PredictMatchingScore(sm_model)

    dataframe_obj = DataPreprocess()
    model_data = dataframe_obj.run_preprocessing_pipeline(RESUME_PATH, JOB_DESCRIPTION_PATH)
    print(model_data.columns)

    model_data = pred_obj.generate_results(model_data)
    print(model_data.columns)
    print(model_data.iloc[0])

