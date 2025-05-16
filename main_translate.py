import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL'] = '1'
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
from process_data import *
from tqdm import tqdm
from config import *
import pickle
import torch

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

def main(model_name, dataset):
    dataset_instance = name_to_dataset_instance(dataset)

    model = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(), max_model_len=3000)
    sampling_params = SamplingParams(temperature=0.1, logprobs=2, max_tokens=300)

    data_output_path = os.path.join(translated_generation_paths, f"{dataset_instance.name}_{model_name[model_name.rfind('/')+1:]}.pkl")
    if os.path.exists(data_output_path):
        with open(data_output_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = dataset_instance.get_train(language="English", num_rows=20000)
    
    for cot_language in tqdm(LANGUAGE_SET):
        # with reasoning
        if f'generated_output_wr_{cot_language}' not in data.keys():
            if "CultureAtlas" in dataset_instance.name:
                data[f'input_wr_{cot_language}'] = data.apply(lambda x: dataset_instance.LANGUAGE_SET_REASONING[cot_language](x), axis=1)
            else:
                data[f'input_wr_{cot_language}'] = data.apply(lambda x: dataset_instance.LANGUAGE_SET_REASONING[cot_language](x), axis=1)

            raw_outputs = model.generate(data[f'input_wr_{cot_language}'], sampling_params=sampling_params)
            outputs = [out.outputs[0].text for out in raw_outputs]
            data[f'generated_output_wr_{cot_language}'] = outputs
            data[f'final_output_{cot_language}'] = [o[o.find('final_answer'):] for o in outputs]

        # without reasoning
        if 'generated_output_wor' not in data.keys():
            data['input_wor'] = data.apply(lambda x: dataset_instance.prompt_wor(x), axis=1) + without_reasoning_json
            raw_outputs = model.generate(data['input_wor'], sampling_params=sampling_params)
            outputs = [out.outputs[0].text for out in raw_outputs]            
            data['generated_output_wor'] = outputs
            data[f'final_output_wor'] = [o[o.find('final_answer'):] for o in outputs]
            
        with open(data_output_path, 'wb+') as f:
            pickle.dump(data, f)

    model = None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="")
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
