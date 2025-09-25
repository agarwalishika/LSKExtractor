from config import *
from glob import glob
import os
import pickle
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from scipy.stats import mode
import hdbscan


def get_clusters(dataset):
    global NUM_CLUSTERS
    centers_file = f"clustering_metadata/clusters_{dataset}_{NUM_CLUSTERS}.pkl"
    assigned_clusters_file = f"clustering_metadata/assigned_clusters_{dataset}_{NUM_CLUSTERS}.pkl"
    if not os.path.exists(centers_file):
        # get inputs
        with open(f'data/{dataset}/data_English.pkl', 'rb') as f:
            df = pickle.load(f)
        inputs = list(df['input'])
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        embeddings = model.encode(inputs, normalize_embeddings=True)
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
        best_lang = ["English"]
        lang_accuracy = [0]
        for lang in datasets.keys():
            df = datasets[lang][assigned_clusters == i]
            acc = sum(df['final_parse_answer'] == df['output']) / (len(df) + 1e-10)

            if acc > min(lang_accuracy):
                lang_accuracy.append(acc)
                best_lang.append(lang)
                if len(lang_accuracy) > 3:
                    min_index = lang_accuracy.index(min(lang_accuracy))
                    lang_accuracy.pop(min_index)
                    best_lang.pop(min_index)
        cluster_langs.append(best_lang)
    return cluster_langs

def main(model_name, dataset, suffix):
    data_files = glob(f"generations_json/{dataset}*{model_name}{suffix}.json")

    train_centers, clustered_training_points = get_clusters(dataset)
    assert len(data_files) == 16

    train_langs = get_langs(clustered_training_points, data_files)

    with open(f'data/{dataset}/data_English.pkl', 'rb') as f:
        test_df = pickle.load(f)[-NUM_TEST:]
    inputs = list(test_df['input'])
    model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
    embeddings = model.encode(inputs, normalize_embeddings=True, batch_size=5)
    del model

    similarities = cosine_similarity(embeddings, train_centers)
    test_assigned_clusters = np.argmax(similarities, axis=1)

    datasets = {}
    for data_file in data_files:
        df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
        datasets[pick_out_lang(data_file)] = df
    
    num_accurate = 0
    for i in range(NUM_CLUSTERS):
        voted_answers = []
        for tlang in train_langs[i]:
            voted_answers.append(list(datasets[tlang][test_assigned_clusters == i].apply(lambda x: let_to_num[x['final_parse_answer'].split('.')[0]], axis=1)))
        if len(voted_answers) == 0: continue
        try:
            majority_answers = mode(np.array(voted_answers), axis=0).mode
            output_numerical_answers = list(df[test_assigned_clusters == i].apply(lambda x: let_to_num[x['output'].split('.')[0]], axis=1))
            num_accurate += sum(majority_answers == output_numerical_answers)
        except:
            continue

    accuracy = num_accurate / NUM_TEST
    with open(f'results_LSKExtractor-Top3({NUM_CLUSTERS}).txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{accuracy}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{accuracy}\n")
        

    
def set_split(temp_dataset):
    global NUM_TEST, NUM_TRAIN
    if "culture" not in temp_dataset:
        NUM_TRAIN = 8000
        NUM_TEST = 2000
    else:
        NUM_TRAIN = 5000
        NUM_TEST = 1533
    
for qa_dataset in EVALUATION_DATASETS:
    set_split(qa_dataset)
    for llm in EVALUATION_LLMS:
        for suffix in ["", "_NR"]:
            for NUM_CLUSTERS in [0]:
                # try:
                    print(f"Model: {llm}, Dataset: {qa_dataset}, NUM_TRAIN: {NUM_TRAIN}, NUM_TEST: {NUM_TEST}")
                    main(llm.replace('/', '_'), qa_dataset, suffix)
                # except:
                #     continue