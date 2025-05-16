from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
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

reasoning_json = lambda lang: f"""Fill out the following JSON:
{{
    "reasoning_in_{lang}": "<your reasoning steps in {lang}>",
    "final_answer": "<output answer here>"
}}
"""

without_reasoning_json = f"""Fill out the following JSON:
{{
    "final_answer": "<output answer here>"
}}
"""


def main(model_name, dataset, eval_config=E5):
    eval_config['load_function'](eval_config)
    dataset_instance: CultureAtlasDataset | BLEnDDataset | SocialIQADataset = name_to_dataset_instance(dataset)
    
    data_output_path = os.path.join(translated_generation_paths, f"{dataset_instance.name}_{model_name[model_name.rfind('/')+1:]}.pkl")
    test_data_output_path = os.path.join(translated_generation_paths, f"TEST_{dataset_instance.name}_{model_name[model_name.rfind('/')+1:]}.pkl")

    if os.path.exists(test_data_output_path):
        with open(test_data_output_path, 'rb') as f:
            test_data = pickle.load(f)
    else:
        ### cluster
        print('\n\n\nClustering...')
        test_data = dataset_instance.get_test(language="English", num_rows=20000)
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        input_texts = list(test_data['input'])
        embeddings = model.encode(input_texts, normalize_embeddings=True)

        centers = dataset_instance.get_train_centers()
        similarities = cosine_similarity(embeddings, centers)
        test_data['assigned_cluster'] = np.argmax(similarities, axis=1)
        del model

        ### generate stuff
        print('\n\n\nGenerating...')
        model = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(), max_model_len=7000)
        sampling_params = SamplingParams(temperature=0.1, logprobs=2, max_tokens=300)

        # with open(data_output_path, 'rb') as f:
        #     train_data = pickle.load(f)
        
        topk_path = data_output_path.split('.pkl')
        topk_path = topk_path[0] + "topk_langs" + ".pkl"
        with open(topk_path, 'rb') as f:
            topk_langs = pickle.load(f)

        # no reasoning
        test_data['input_wor'] = test_data.apply(lambda x: dataset_instance.prompt_wor(x), axis=1) + without_reasoning_json
        raw_outputs = model.generate(test_data['input_wor'], sampling_params=sampling_params)
        test_data[f'baseline_wor'] = [out.outputs[0].text for out in raw_outputs]
        with open(test_data_output_path, 'wb+') as f:
            pickle.dump(test_data, f)

        # only English reasoning
        test_data['input_wr_English'] = [dataset_instance.LANGUAGE_SET_REASONING["English"](x) for _, x in test_data.iterrows()]
        raw_outputs = model.generate(test_data['input_wr_English'], sampling_params=sampling_params)
        test_data[f'baseline_English'] = [out.outputs[0].text for out in raw_outputs]

        with open(test_data_output_path, 'wb+') as f:
            pickle.dump(test_data, f)
        # top-1 MLKExtractor
        test_data['lang'] = test_data['assigned_cluster'].apply(lambda x: topk_langs[x][0].split('_')[-1])
        test_data['input_wr_mlk'] = test_data.apply(lambda x: dataset_instance.LANGUAGE_SET_REASONING[x['lang']](x), axis=1)   
        raw_outputs = model.generate(test_data['input_wr_mlk'], sampling_params=sampling_params)
        test_data[f'baseline_mlk'] = [out.outputs[0].text for out in raw_outputs]
        del model
        
        with open(test_data_output_path, 'wb+') as f:
            pickle.dump(test_data, f)
        
    ### evaluate
    print("\n\n\nEvaluating...")
    model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    test_data['scores_wor'] = dataset_instance.get_scores(model, test_data, 'baseline_wor')
    test_data['scores_English'] = dataset_instance.get_scores(model, test_data, 'baseline_English')
    test_data['scores_mlk'] = dataset_instance.get_scores(model, test_data, 'baseline_mlk')
    mean_wr = test_data[f'scores_wor'].mean()
    mean_English = test_data[f'scores_English'].mean()
    mean_mlk = test_data[f'scores_mlk'].mean()
    with open(dataset_instance.name + "_test_results.txt", 'a+') as f:
        f.write(f"{dataset_instance.name}, {model_name}, {mean_wr}, {mean_English}, {mean_mlk}\n")
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", type=str, default="")
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
