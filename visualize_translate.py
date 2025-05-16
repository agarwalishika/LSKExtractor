import os
os.environ['VLLM_ALLOW_LONG_MAX_MODEL'] = '1'
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from config import *
import numpy as np
import pickle
import torch

def evaluate(store_file_name):
    country_store = {}
    countries = []
    
    # load models
    # configuration['load_function'](configuration)

    files = glob(translated_generation_paths + f"*{dataset_name}*")

    for file in tqdm(files):
        is_visualize = ("aya-23-8B" in file or "Phi-3-small-8" in file or "gemma-3-1b" in file) and ("BLEnD" in file or "CultureAtlas" in file) and ("topk" not in file and "TEST" not in file)
        if not is_visualize:
            continue
        code = file[file.rfind('/') + 1:file.rfind('.')]
        
        with open(file, 'rb') as f:
            dataset = pickle.load(f)

        for lang in LANGUAGE_SET:
            if "BLEnD" in file and "aya-23-8B" in file:
                hi = 9
            try:
                without_reasoning_scores = np.array(torch.Tensor(list(dataset['scores_wor'])))
                dataset['only_with'] = torch.Tensor(list(dataset[f'scores_wr_{lang}']))
                dataset['difference'] = (dataset['only_with'] - without_reasoning_scores)

                temp = np.array(list(dataset.groupby([key])['scores_wor'].mean()))
                country_scores = {"only_with": list(dataset.groupby([key])['only_with'].mean()), 
                                    "difference": list(dataset.groupby([key])['difference'].mean()),
                                    "percent_change": list(dataset.groupby([key])['difference'].mean()) / (np.array(list(dataset.groupby([key])['scores_wor'].mean())) + 1e-10)}
                
                countries = list(dataset.groupby([key])['only_with'].mean().index)

                # i = LANGUAGE_SET.index(code.split('_')[-1])
                model  = code.split('_')[1]
                if model not in country_store.keys():
                    country_store[model] = []
                country_store[model].append(country_scores)
            except:
                print('ERROR with', file)
    with open(store_file_name, 'wb+') as f:
        pickle.dump((country_store, countries), f)

def visualize(store_file_name, setting):
    with open(store_file_name, 'rb') as f:
        country_store, countries = pickle.load(f)
    
    for model in country_store.keys():
        if len(country_store[model]) == 0:
            continue
        data = np.array([x[setting] for x in country_store[model][:len(LANGUAGE_SET)]]).transpose()

        fig, ax = plt.subplots(figsize=(12, 15))
        im = ax.imshow(data)

        font_size = 15 if len(countries) > 40 else 20
        ax.set_xticks(range(len(LANGUAGE_SET)), labels=LANGUAGE_SET,
                    rotation=45, ha="right", va="baseline", rotation_mode="anchor", fontsize=font_size)
        ax.set_yticks(range(len(countries)), labels=list(countries), fontsize=font_size)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")

        plt.title(model, fontsize=20)
        plt.xlabel('Query Language', fontsize=20)
        plt.ylabel('Country', fontsize=20)
        plt.savefig(f"figures_{setting}/{dataset_name}_{key}_{model}.png",bbox_inches='tight')
        plt.clf()
        plt.close()

if __name__ == "__main__":
    E5 = None
    settings = ["difference"]

    # visualize clusters
    key = 'assigned_cluster'

    # -------------------------------------------------------------------------------------------------------
    dataset_name = "CultureAtlas"

    store_file_name_main = os.path.join(translated_generation_paths, f"{dataset_name}_visualization_store.pkl")
    evaluate(E5, store_file_name_main)
    for setting in settings:
        visualize(store_file_name_main, setting)
    

    # -------------------------------------------------------------------------------------------------------
    dataset_name = "BLEnD"

    store_file_name_main = os.path.join(translated_generation_paths, f"{dataset_name}_visualization_store.pkl")
    evaluate(E5, store_file_name_main)
    for setting in settings:
        visualize(store_file_name_main, setting)
    
    # -------------------------------------------------------------------------------------------------------
    print('\n\n\n\nHELLO\n\n\n\n')
    dataset_name = "SocialIQA"

    store_file_name_main = os.path.join(translated_generation_paths, f"{dataset_name}_visualization_store.pkl")
    evaluate(E5, store_file_name_main)
    for setting in settings:
        visualize(store_file_name_main, setting)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # visualize by country
    key = 'country'

    # -------------------------------------------------------------------------------------------------------
    dataset_name = "CultureAtlas"

    store_file_name_main = os.path.join(translated_generation_paths, f"{dataset_name}_visualization_store.pkl")
    evaluate(E5, store_file_name_main)
    for setting in settings:
        visualize(store_file_name_main, setting)
    

    # -------------------------------------------------------------------------------------------------------
    dataset_name = "BLEnD"

    store_file_name_main = os.path.join(translated_generation_paths, f"{dataset_name}_visualization_store.pkl")
    evaluate(E5, store_file_name_main)
    for setting in settings:
        visualize(store_file_name_main, setting)
    
