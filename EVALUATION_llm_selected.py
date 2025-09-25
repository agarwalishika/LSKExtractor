import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from config import *
from glob import glob
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from config import *
import pickle
from glob import glob
import torch

LANG_SELECTION_INSTRUCTION = lambda row: f"""An expert language is the language from the provided list that is most appropriate and informative for answering the given question (e.g., because the question is about a culture, region, or source where that language is dominant, or because that language has the richest knowledge base for the topic).

From the following languages:

[Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, Hindi, Turkish, Bengali]

determine which one is the best expert language for answering the question below.

Question: {row['input']}
Fill out your language expert in the below JSON format:
{{
    "expert_language": "<the expert language from the above list>"
}}
"""

save_file_name = lambda dataset, model: f"clustering_metadata/llm_selected_langs_{dataset}_{model.replace('/', '_')}.pkl"

def parse_languages(outputs):
    languages = []
    for i, output in enumerate(outputs):
        for lang in LANGUAGE_SET:
            if lang in output:
                languages.append(lang)
                break
        if len(languages) == i:
            languages.append("English")
    return languages

def get_languages(model_name, dataset):
    if not os.path.exists(save_file_name(dataset, model_name)):
        # generations
        language_model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=512, top_p=0.9)

        with open(f'data/{dataset}/data_English.pkl', 'rb') as f:
            translated_df = pickle.load(f)[-NUM_TEST:]
        # translated_df = translated_df[~translated_df['input'].isna()]
        # translated_df = translated_df[~translated_df['choices'].isna()]
        
        input_queries = list(translated_df.apply(lambda row: LANG_SELECTION_INSTRUCTION(row), axis=1))
        outputs = language_model.generate(input_queries, sampling_params=sampling_params)
        outputs = [out.outputs[0].text for out in outputs]
        languages = parse_languages([o[o.find('expert_language'):].split('\n')[0] for o in outputs])

        with open(save_file_name(dataset, model_name), 'wb+') as f:
            pickle.dump(languages, f)
        
        del language_model, sampling_params
    else:
        with open(save_file_name(dataset, model_name), 'rb') as f:
            languages = pickle.load(f)
    
    return languages

def main(model_name, dataset, suffix):
    languages = np.array(get_languages(model_name, dataset))

    model_name = model_name.replace('/', '_')
    data_files = glob(f"generations_json/{dataset}*{model_name}{suffix}.json")
    assert len(data_files) == 16

    num_acccurate = 0
    for data_file in data_files:
        df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
        df_lang = pick_out_lang(data_file)
        df = df[languages == df_lang]
        num_acccurate += sum(df['final_parse_answer'] == df['output'])

    with open('results_LLMSelected.txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{num_acccurate / NUM_TEST}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{num_acccurate / NUM_TEST}\n")
    
for qa_dataset in reversed(EVALUATION_DATASETS):
    for llm in (EVALUATION_LLMS):
        for suffix in ["", "_NR"]:
            # try:
                print(f"Model: {llm}, Dataset: {qa_dataset}")
                main(llm, qa_dataset, suffix)
            # except:
            #     continue
