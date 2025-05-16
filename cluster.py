from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from process_data import *
from glob import glob
from tqdm import tqdm
from config import *
import pickle
import os

def main():
    files = glob(translated_generation_paths + "*")
    for data_output_path in tqdm(files):
        if "visualization_store" in data_output_path or "topk_langs" in data_output_path or "TEST" in data_output_path or os.path.isdir(data_output_path):
            continue
        if os.path.exists(data_output_path):
            with open(data_output_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = dataset_instance.get_train(language="English", num_rows=20000)
        
        topk_path = data_output_path.split('.pkl')
        topk_path = topk_path[0] + "topk_langs" + ".pkl"
        if os.path.exists(topk_path):
            continue
        
        dataset_instance = name_to_dataset_instance(data_output_path.split('/')[-1].split('_')[0])
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        input_texts = list(data['input'])
        embeddings = model.encode(input_texts, normalize_embeddings=True)

        centers = dataset_instance.get_train_centers()
        similarities = cosine_similarity(embeddings, centers)
        data['assigned_cluster'] = np.argmax(similarities, axis=1)

        lang_keys = [key for key in data.keys() if "scores_wr" in key]
        if len(lang_keys) == 0:
            continue
        mean_cluster_scores = data[lang_keys + ['assigned_cluster']].groupby('assigned_cluster').mean()
        
        for i in range(len(centers)):
            if i not in mean_cluster_scores.index:
                new_row = pd.DataFrame([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], index=[i], columns=mean_cluster_scores.columns)
                mean_cluster_scores = pd.concat([mean_cluster_scores, new_row])
        mean_cluster_scores = mean_cluster_scores.sort_index()
        top_k_langs = list(mean_cluster_scores.apply(lambda s: s.abs().nlargest(3).index.tolist(), axis=1))

        with open(topk_path, 'wb+') as f:
            pickle.dump(top_k_langs, f)

        with open(data_output_path, 'wb+') as f:
            pickle.dump(data, f)

if __name__ == '__main__':    
    main()