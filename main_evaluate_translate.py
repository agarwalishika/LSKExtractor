from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from process_data import *
from tqdm import tqdm
from config import *
import pickle
import torch
import os

embedding_model = None

def load_ST(config):
    global embedding_model
    embedding_model = SentenceTransformer(config['model_name'])

def get_ST_similarity(generated, ground_truth):
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given two documents, calculate the similarity between the two documents.'
    queries = [
        get_detailed_instruct(task, prompt) for prompt in list(ground_truth)
    ]
    # No need to add instruction for retrieval documents
    documents = list(generated)

    embeddings_1 = embedding_model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    embeddings_2 = embedding_model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
    scores = torch.sum((embeddings_1 * embeddings_2), axis=1).cpu()

    return scores

E5 = {"model_name": "intfloat/multilingual-e5-large-instruct", "file_suffix": "e5", "load_function": load_ST, "sim_function": get_ST_similarity}
GEMMA2 = {"model_name": "BAAI/bge-multilingual-gemma2", "file_suffix": "gemma", "load_function": load_ST, "sim_function": get_ST_similarity}


def main(model_name, dataset, eval_config=E5):
    eval_config['load_function'](eval_config)
    dataset_instance: CultureAtlasDataset | BLEnDDataset | SocialIQADataset = name_to_dataset_instance(dataset)

    data_output_path = os.path.join(translated_generation_paths, f"{dataset_instance.name}_{model_name[model_name.rfind('/')+1:]}.pkl")
    if os.path.exists(data_output_path):
        with open(data_output_path, 'rb') as f:
            data = pickle.load(f)
    
    for cot_language in tqdm(LANGUAGE_SET):        
        # evaluate
        if 'scores_wor' not in data.keys():
            data = data[data[f'generated_output_wor'].notna()]
            data['scores_wor'] = dataset_instance.get_scores(embedding_model, data, 'generated_output_wor')
        
        if f'scores_wr_{cot_language}' not in data.keys():
            data = data[data[f'generated_output_wr_{cot_language}'].notna()]
            data[f'scores_wr_{cot_language}'] = dataset_instance.get_scores(embedding_model, data, f'generated_output_wr_{cot_language}')

    with open(data_output_path, 'wb+') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini-IB")
    parser.add_argument("--dataset", type=str, default="SocialIQA")
    args = parser.parse_args()

    # specific model and dataset
    if len(args.model) > 0 and len(args.dataset) > 0:
        main(args.model, args.dataset)

    # all models, specific dataset
    elif len(args.dataset) > 0:
        for model_name in model_names:
            main(model_name, args.dataset)
    
    # specific model, all datasets
    elif len(args.model) > 0:
        for dataset in datasets:
            main(args.model, dataset)
    
    # all models, all datasets
    else:
        for model_name in model_names:
            for dataset in datasets:
                main(model_name, dataset)