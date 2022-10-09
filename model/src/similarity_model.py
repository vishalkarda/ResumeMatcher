import pickle

import joblib

from .data_preprocessing import DataPreprocess
from .data_preprocessing import RESUME_PATH, JOB_DESCRIPTION_PATH
from .utils import get_destination_path


class SimilarityModel:

    @staticmethod
    def __load_model(model_name):
        """"""
        # model = SentenceTransformer(model_name)
        # output_path = get_destination_path("data\\trained_model")
        # joblib.dump(model, output_path + "\\" + str("bert_transformer"))

        # model_path = get_destination_path("\\model\\data\\trained_model")
        # model_path = get_destination_path("/model/data/trained_model")
        #
        # # model = joblib.load(model_path + "\\"+model_name)
        # print("Path", model_path + "/" + model_name)
        model = model_name

        with open('bert_transformer.pkl', 'rb') as f:
            model = pickle.load(f)

        return model

    def get_model(self, model_name='bert_transformer.pkl'):
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
        # cos_sim_path = get_destination_path("\\model\\src")
        # cos_sim = joblib.load(cos_sim_path + "\\" + "cos_sim")

        # cos_sim_path = get_destination_path("/model/src")
        # cos_sim = joblib.load(cos_sim_path + "/" + "cos_sim")

        with open('cos_sim.pkl', 'rb') as f:
            cos_sim = pickle.load(f)

        cosine_scores = cos_sim(embeddings1, embeddings2)

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
    pred_obj = PredictMatchingScore(sm_model)

    dataframe_obj = DataPreprocess()
    model_data = dataframe_obj.run_preprocessing_pipeline(RESUME_PATH, JOB_DESCRIPTION_PATH)

    model_data = pred_obj.generate_results(model_data)


