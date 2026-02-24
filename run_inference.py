from vllm import LLM, SamplingParams
from config import *
import pickle
import os
from glob import glob
import torch

save_file_name = lambda dataset, model, language: f"generations/{dataset}_{language}_{model.replace('/', '_')}.pkl"

os.makedirs('generations', exist_ok=True)

for model_n in EVALUATION_LLMS:
        language_model = LLM(model=model_n, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=512, top_p=0.9)

        for dataset in ["blend", "culture_atlas", "social_iqa"]:
            for language in LANGUAGE_SET:
                save_file = save_file_name(dataset, model_n, language)
                if os.path.exists(save_file):
                    print(f"Skipping {save_file}, already exists.")
                    continue
            
                with open(f'data/{dataset}/data_{language}.pkl', 'rb') as f:
                    translated_df = pickle.load(f)
                
                input_queries = list(translated_df.apply(lambda row: TRANSLATED_INSTRUCTIONS[language](row), axis=1))
                outputs = language_model.generate(input_queries, sampling_params=sampling_params)
                outputs = [out.outputs[0].text for out in outputs]
                translated_df['vllm_response'] = outputs

                with open(save_file, 'wb+') as f:
                    pickle.dump(translated_df, f)
        
        del language_model, sampling_params