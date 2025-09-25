import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from vllm import LLM, SamplingParams
from config import *
import pickle
from glob import glob
import torch

EVALUATION_LLMS = [
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "CohereLabs/aya-23-8B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]


save_file_name_nr = lambda dataset, model, language: f"generations/{dataset}_{language}_{model.replace('/', '_')}_NR.pkl"
save_file_name = lambda dataset, model, language: f"generations/{dataset}_{language}_{model.replace('/', '_')}.pkl"

for model_n in (EVALUATION_LLMS):
        language_model = LLM(model=model_n, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=512, top_p=0.9)

        for dataset in ["blend", "culture_atlas", "social_iqa"]:
            for language in LANGUAGE_SET:
                ######################### NO REASONING ###############################
                save_file = save_file_name_nr(dataset, model_n, language)
                if os.path.exists(save_file):
                    print(f"Skipping {save_file}, already exists.")
                    continue
            
                print(f"Generating {save_file}...")
                with open(f'data/{dataset}/data_{language}.pkl', 'rb') as f:
                    translated_df = pickle.load(f)

                if "blend" in dataset:
                    translated_df['choices_possibilities'] = ["A/B/C/D"] * len(translated_df)
                if "culture" in dataset:
                    translated_df['choices_possibilities'] = ["A/B"] * len(translated_df)
                if "social" in dataset:
                    translated_df['choices_possibilities'] = ["A/B/C"] * len(translated_df)
                
                
                input_queries = list(translated_df.apply(lambda row: NR_TRANSLATED_INSTRUCTIONS[language](row), axis=1))
                outputs = language_model.generate(input_queries, sampling_params=sampling_params)
                outputs = [out.outputs[0].text for out in outputs]
                translated_df['vllm_response'] = outputs

                with open(save_file, 'wb+') as f:
                    pickle.dump(translated_df, f)                
                ######################### NO REASONING ###############################



            for language in LANGUAGE_SET:
                ########################### REASONING #################################
                save_file = save_file_name(dataset, model_n, language)
                if os.path.exists(save_file):
                    print(f"Skipping {save_file}, already exists.")
                    continue

                print(f"Generating {save_file}...")
                with open(f'data/{dataset}/data_{language}.pkl', 'rb') as f:
                    translated_df = pickle.load(f)

                input_queries = list(translated_df.apply(lambda row: TRANSLATED_INSTRUCTIONS[language](row), axis=1))
                outputs = language_model.generate(input_queries, sampling_params=sampling_params)
                outputs = [out.outputs[0].text for out in outputs]
                translated_df['vllm_response'] = outputs

                with open(save_file, 'wb+') as f:
                    pickle.dump(translated_df, f)
                ########################### REASONING #################################
        
        del language_model, sampling_params