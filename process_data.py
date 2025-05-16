from sentence_transformers import SentenceTransformer
from sklearn.cluster import kmeans_plusplus
from datasets import load_dataset
from config import CultureAtlas_LANGUAGE_SET_REASONING, MCQ_LANGUAGE_SET_REASONING
import pandas as pd
import numpy as np
import pickle
import os

######### CultureAtlas DATASET #########
class CultureAtlasDataset:
    def __init__(self):
        self.name = "CultureAtlas"
        self.prompt_wr = lambda cot_lang, row: f"""Is the following correct? {row['input']}\nThink about and answer in {cot_lang}, and then output our final answer as True or False."""
        self.prompt_wor = lambda row: f"""Is the following correct about {row['country']}'s {row['topic']} (specifically, {row['subsubtopic']})? {row['input']}. Output your final answer as True or False."""
        
        self.countries = ["United States of America", "India", "Nigeria", "South Africa", "Brazil", "Indonesia", "China", "Mexico", "Canada", "France", "Pakistan", "Turkey", "Russia", "United Arab Emirates", "Japan"]
        self.LANGUAGE_SET_REASONING = CultureAtlas_LANGUAGE_SET_REASONING

    def get_train_centers(self):
        center_file = 'mlk_centers/culture_atlas.pkl'
        if not os.path.exists(center_file):
            data = self.get_train()
            model = SentenceTransformer('intfloat/multilingual-e5-large')
            input_texts = list(data['input'])
            embeddings = model.encode(input_texts, normalize_embeddings=True)

            centers, indices = kmeans_plusplus(embeddings, n_clusters=48, random_state=512)
            with open(center_file, 'wb+') as f:
                pickle.dump(centers, f)
        
        with open(center_file, 'rb') as f:
            centers = pickle.load(f)
        return centers
            

    def get_culture_atlas(self, split, language, num_rows=None):
        with open('data/culture_atlas/processed_dataframe.pkl', 'rb') as f:
            df: pd.DataFrame = pickle.load(f)

        df = df[df['country'].isin(self.countries)]

        # split data into train and test
        df = df.sample(frac=1.0, random_state=512)
        train_size = int(len(df) * 0.7)
        if split == "train":
            df = df.iloc[:train_size]
        else:
            df = df.iloc[train_size:]

        # cut dataset if necessary
        if num_rows is not None:
            return df.iloc[:num_rows]
        return df

    def get_train(self, language=None, num_rows=None):
        return self.get_culture_atlas("train", language, num_rows)

    def get_test(self, language=None, num_rows=None):
        return self.get_culture_atlas("test", language, num_rows)

    def get_scores(self, model, data, generated_key):
        scores = []

        true_str = "true"
        false_str = "false"
        
        input_embedding = model.encode(list(data[generated_key]))
        true_embedding = model.encode(true_str)
        false_embedding = model.encode(false_str)
        
        # Compute cosine similarity between the input and "true", and input and "false"
        similarity_true = np.dot(input_embedding, true_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(true_embedding))
        similarity_false = np.dot(input_embedding, false_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(false_embedding))
        predicted = similarity_true > similarity_false
        
        scores = np.zeros(len(predicted))
        scores[predicted == np.array(data['output'])] = 1

        return np.array(scores)


######### CultureAtlas DATASET #########


######### BLEnD DATASET #########
#https://proceedings.neurips.cc/paper_files/paper/2024/file/8eb88844dafefa92a26aaec9f3acad93-Paper-Datasets_and_Benchmarks_Track.pdf
class BLEnDDataset:
    def __init__(self):
        self.name = "BLEnD"
        #["English", "Chinese", "Spanish", "Indonesian", "Korean", "Greek", "Persian", "Arabic", "Azerbaijani", "Sudanese", "Assamese", "Hausa", "Amharic"]
        self.prompt_wr = lambda cot_lang, row: f"Question: {row['input']}\nFirst, reason in {cot_lang}, then choose one and only one of the following answer choices:\n{row['choices']}."
        self.prompt_wor = lambda row: f"Question: {row['input']}\nChoose one and output only one of the following answer choices:\n{row['choices']}."

        self.LANGUAGE_SET_REASONING = MCQ_LANGUAGE_SET_REASONING
    
    def get_train_centers(self):
        center_file = 'mlk_centers/blend.pkl'
        if not os.path.exists(center_file):
            data = self.get_train()
            model = SentenceTransformer('intfloat/multilingual-e5-large')
            input_texts = list(data['input'])
            embeddings = model.encode(input_texts, normalize_embeddings=True)

            centers, indices = kmeans_plusplus(embeddings, n_clusters=48, random_state=512)
            with open(center_file, 'wb+') as f:
                pickle.dump(centers, f)
        
        with open(center_file, 'rb') as f:
            centers = pickle.load(f)
        return centers

    def get_blend(self, split, num_rows=None):
        with open('data/blend/processed_blend_data.pkl', 'rb') as f:
            df = pickle.load(f)

        train_size = int(len(df) * 0.7)
        if split == "train":
            df = df.iloc[:train_size]
        else:
            df = df.iloc[train_size:]

        # cut dataset if necessary
        if num_rows is not None:
            return df.iloc[:num_rows]
    
        return df

    def get_train(self, language=None, num_rows=None):
        return self.get_blend("train", num_rows)

    def get_test(self, language=None, num_rows=None):
        return self.get_blend("test", num_rows)
    
    def get_scores(self, model, data, generated_key):
        scores = []
        input_embeddings = model.encode(list(data[generated_key]))
        choice_strs = list(data['choices'].apply(lambda x: x.split("\n")))

        if 'choice_embs' not in data.keys():
            data['choice_embs'] = list(data['choices'].apply(lambda x: np.array([model.encode(cs) for cs in x.split("\n")])))
            
        choice_embs = data['choice_embs']

        i = 0
        for input_embedding, choice_str, choice_emb in zip(input_embeddings, choice_strs, choice_embs):
            similarities = np.dot(choice_emb, input_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(choice_emb))
            scores.append(int(choice_str[np.argmax(similarities)] == data.iloc[i]['output']))
            i += 1

        return np.array(scores)

######### BLEnD DATASET #########


######### SocialIQA DATASET #########
# https://aclanthology.org/D19-1454.pdf
class SocialIQADataset:
    def __init__(self):
        self.name = "SocialIQA"

        self.prompt_wr = lambda cot_lang, row: f"Answer the question about the appropriate social norm. {row['input']}\nFirst, reason in {cot_lang}, then choose one and only one of the following answer choices:\n{row['choices']}."
        self.prompt_wor = lambda row: f"Answer the question about the appropriate social norm. {row['input']}\nChoose one and output only one of the following answer choices:\n{row['choices']}."

        self.LANGUAGE_SET_REASONING = MCQ_LANGUAGE_SET_REASONING
    
    def get_train_centers(self):
        center_file = 'mlk_centers/socialiqa.pkl'
        if not os.path.exists(center_file):
            data = self.get_train()
            model = SentenceTransformer('intfloat/multilingual-e5-large')
            input_texts = list(data['input'])
            embeddings = model.encode(input_texts, normalize_embeddings=True)

            centers, indices = kmeans_plusplus(embeddings, n_clusters=48, random_state=512)
            with open(center_file, 'wb+') as f:
                pickle.dump(centers, f)
        
        with open(center_file, 'rb') as f:
            centers = pickle.load(f)
        return centers

    def get_socialiqa(self, split, num_rows=None):
        with open('data/social_iqa/processed_social_iqa_data.pkl', 'rb') as f:
            df = pickle.load(f)

        train_size = int(len(df) * 0.7)
        if split == "train":
            df = df.iloc[:train_size]
        else:
            df = df.iloc[train_size:]

        # cut dataset if necessary
        if num_rows is not None:
            return df.iloc[:num_rows]
    
        return df

    def get_train(self, language=None, num_rows=None):
        return self.get_socialiqa("train", num_rows)

    def get_test(self, language=None, num_rows=None):
        return self.get_socialiqa("test", num_rows)
    
    def get_scores(self, model, data, generated_key):
        scores = []
        input_embeddings = model.encode(list(data[generated_key]))
        choice_strs = list(data['choices'].apply(lambda x: [y[2:].strip() for y in x.split("\n")]))

        if 'choice_embs' not in data.keys():
            data['choice_embs'] = list(data['choices'].apply(lambda x: np.array([model.encode(cs) for cs in x.split("\n")])))
            
        choice_embs = data['choice_embs']

        i = 0
        for input_embedding, choice_str, choice_emb in zip(input_embeddings, choice_strs, choice_embs):
            similarities = np.dot(choice_emb, input_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(choice_emb))
            scores.append(int(choice_str[np.argmax(similarities)] == data.iloc[i]['output']))
            i += 1

        return np.array(scores)

######### SocialIQA DATASET #########


def name_to_dataset_instance(name):
    datasets = [CultureAtlasDataset, BLEnDDataset, SocialIQADataset]
    for set in datasets:
        if name == set().name:
            return set()
    return "ERROR - NO DATASET OF THAT NAME WAS FOUND"