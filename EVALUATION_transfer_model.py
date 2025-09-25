import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from config import *
from glob import glob
import pickle
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import hdbscan

def get_clusters(dataset):
    global NUM_CLUSTERS
    centers_file = f"clustering_metadata/clusters_{dataset}_{NUM_CLUSTERS}.pkl"
    assigned_clusters_file = f"clustering_metadata/assigned_clusters_{dataset}_{NUM_CLUSTERS}.pkl"
    if not os.path.exists(centers_file):
        # get inputs
        with open(f'data/{dataset}/data_English.pkl', 'rb') as f:
            df = pickle.load(f)
        inputs = list(df['input'][:NUM_TRAIN])
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        embeddings = model.encode(inputs, normalize_embeddings=True, batch_size=10)
        del model

        # cluster
        if NUM_CLUSTERS > 0:
            centers, _ = kmeans_plusplus(embeddings, n_clusters=NUM_CLUSTERS, random_state=512)
            with open(centers_file, 'wb+') as f:
                pickle.dump(centers, f)
        else:
            # Cluster with HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,      # Minimum size of clusters
                min_samples=5,           # Minimum samples for core points
                metric='euclidean'       # Can also try 'cosine'
            )

            cluster_labels = clusterer.fit_predict(embeddings)
            cluster_labels -= min(cluster_labels)

            # Extract cluster centers (mean embedding of points in each cluster)
            unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
            centers = []
            for label in unique_labels:
                cluster_points = embeddings[cluster_labels == label]
                centers.append(cluster_points.mean(axis=0))
            centers = np.array(centers)

            with open(centers_file, 'wb+') as f:
                pickle.dump(centers, f)
        
        similarities = cosine_similarity(embeddings, centers)
        assigned_clusters = np.argmax(similarities, axis=1)
        with open(assigned_clusters_file, 'wb+') as f:
            pickle.dump(assigned_clusters, f)

    else:
        with open(centers_file, 'rb') as f:
            centers = pickle.load(f)
    
        with open(assigned_clusters_file, 'rb') as f:
            assigned_clusters = pickle.load(f)

    # return the clustering
    NUM_CLUSTERS = len(centers)
    return centers, assigned_clusters

def get_langs(assigned_clusters, data_files):
    datasets = {}
    for data_file in data_files:
        df = pd.read_json(open(data_file, 'r'))[:NUM_TRAIN]
        datasets[pick_out_lang(data_file)] = df

    cluster_langs = []
    for i in range(NUM_CLUSTERS):
        best_lang = "English"
        lang_accuracy = 0
        for lang in datasets.keys():
            df = datasets[lang][assigned_clusters == i]
            acc = sum(df['final_parse_answer'] == df['output']) / (len(df) + 1e-10)

            if acc > lang_accuracy:
                lang_accuracy = acc
                best_lang = lang
        cluster_langs.append(best_lang)
    return cluster_langs

def main(model_name, dataset, suffix):
    data_files = glob(f"generations_json/{dataset}*Llama-3.1-8B-Instruct{suffix}.json")

    train_centers, clustered_training_points = get_clusters(dataset)
    assert len(data_files) == 16

    train_langs = get_langs(clustered_training_points, data_files)
    with open(f'clustering_metadata/cluster_langs_lsk_{dataset}_{NUM_CLUSTERS}|{model_name}.pkl', 'wb+') as f:
        pickle.dump(train_langs, f)

    embeddings_file_name = f'clustering_metadata/{dataset}_test_embeddings.pkl'

    with open(f'data/{dataset}/data_English.pkl', 'rb') as f:
        test_df = pickle.load(f)[-NUM_TEST:]
    inputs = list(test_df['input'])
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
    embeddings = model.encode(inputs, normalize_embeddings=True, batch_size=1)
    del model

    similarities = cosine_similarity(embeddings, train_centers)
    test_assigned_clusters = np.argmax(similarities, axis=1)

    data_files = glob(f"generations_json/{dataset}*{model_name}*{suffix}.json")
    datasets = {}
    for data_file in data_files:
        df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
        datasets[pick_out_lang(data_file)] = df
    
    num_accurate = 0
    for i in range(NUM_CLUSTERS):
        df = datasets[train_langs[i]][test_assigned_clusters == i]
        num_accurate += sum(df['final_parse_answer'] == df['output'])

    accuracy = num_accurate / NUM_TEST
    with open(f'results_transfer_model.txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{accuracy}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{accuracy}\n")
        


NUM_TRAIN = 5000
NUM_TEST = 1533

for llm in EVALUATION_LLMS:
    for suffix in ["", "_NR"]:
        for NUM_CLUSTERS in [12]:
        # try:
            print(f"Model: {llm}, Dataset: culture, NUM_TRAIN: {NUM_TRAIN}, NUM_TEST: {NUM_TEST}")
            main(llm.replace('/', '_'), "culture_atlas", suffix)
        # except:
        #     continue