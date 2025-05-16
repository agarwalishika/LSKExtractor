import os
from process_data import *
from language_codes import name_to_object
from bge_evaluation import evaluate_bge
import pickle
from tqdm import tqdm
from config import *
from argparse import ArgumentParser
import openai
from parallel_inference import InferenceEngine, Hyperparameters
import time

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

    conn = {
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "base_url": os.environ.get("AZURE_OPENAI_BASE_URL"),
        "api_version": "2025-03-01-preview",
    }

    engine = InferenceEngine(
        inference_strategy="azure_openai",
        connection_details=conn,
        model_name=model_name,
    )

    hyperparams = Hyperparameters(temperature=0.1, max_tokens=300)

    dataset_instance = name_to_dataset_instance(dataset)


    data_output_path = os.path.join(translated_generation_paths, f"{dataset_instance.name}_{model_name[model_name.rfind('/')+1:]}_Parallel.pkl")
    print(f"data_output_path: {data_output_path}")
    if os.path.exists(data_output_path):
        with open(data_output_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = dataset_instance.get_train(language="English", num_rows=instance_count)

    instance_count = len(data)
    batch_size = 100
    for cot_language in LANGUAGE_SET:

        lang_start_time = time.time()
        # with reasoning
        if f'generated_output_wr_{cot_language}' not in data.keys():
            print(f"Generating with reasoning in {cot_language}")
            
            if "CultureAtlas" in dataset_instance.name:
                data[f'input_wr_{cot_language}'] = data.apply(lambda x: dataset_instance.LANGUAGE_SET_REASONING[cot_language](x), axis=1)
            else:
                data[f'input_wr_{cot_language}'] = data.apply(lambda x: dataset_instance.LANGUAGE_SET_REASONING[cot_language](x), axis=1)

            
            inputs = [[{"role": "user", "content": input}] for input in data[f'input_wr_{cot_language}']]

            outputs = [None] * instance_count
            for i in range(0, instance_count, batch_size):
                batch = inputs[i:i+batch_size]
                start = time.time()
                outputs[i:i+batch_size] = engine.parallel_messages_inference(batch, hyperparams)
                end = time.time()
                print(f"Time taken for {i//batch_size + 1}th batch: {(end - start) / 60:.2f} minutes")
                print(f"{i//batch_size + 1}th periodic save to {data_output_path}")
                data[f'generated_output_wr_{cot_language}'] = outputs
                data[f'final_output_{cot_language}'] = [o[o.find('final_answer'):] if o else None for o in outputs]
                with open(data_output_path, 'wb+') as f:
                    pickle.dump(data, f)

                # print(f"sleeping for 30 seconds")
                # time.sleep(30)

                data[f'generated_output_wr_{cot_language}'] = outputs
                data[f'final_output_{cot_language}'] = [o[o.find('final_answer'):] if o else None for o in outputs]
                with open(data_output_path, 'wb+') as f:
                    pickle.dump(data, f)

        # # without reasoning
        if 'generated_output_wor' not in data.keys():
            print(f"Generating without reasoning")
            data['input_wor'] = data.apply(lambda x: dataset_instance.prompt_wor(x), axis=1) + without_reasoning_json

            inputs = [[{"role": "user", "content": input}] for input in data['input_wor']]

            outputs = [None] * instance_count
            for i in range(0, instance_count, batch_size):
                batch = inputs[i:i+batch_size]
                start = time.time()
                outputs[i:i+batch_size] = engine.parallel_messages_inference(batch, hyperparams)
                end = time.time()
                print(f"Time taken for {i//batch_size + 1}th batch: {(end - start) / 60:.2f} minutes")
                print(f"{i//batch_size + 1}th periodic save to {data_output_path}") 
                data['generated_output_wor'] = outputs
                data[f'final_output_{cot_language}'] = [o[o.find('final_answer'):] if o else None for o in outputs]
                with open(data_output_path, 'wb+') as f:
                    pickle.dump(data, f)
            
            data['generated_output_wor'] = outputs
            data[f'final_output_wor'] = [o[o.find('final_answer'):] for o in outputs]
            with open(data_output_path, 'wb+') as f:
                pickle.dump(data, f)

        print(f"Finished for language {cot_language} in {dataset_instance.name} with model {model_name}")
        print(f"saving to {data_output_path}")
        with open(data_output_path, 'wb+') as f:
            pickle.dump(data, f)

        lang_end_time = time.time()
        print(f"Time taken for {cot_language}: {(lang_end_time - lang_start_time) / 60:.2f} minutes")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini-IB")
    parser.add_argument("--dataset", type=str, default="SocialIQA")
    args = parser.parse_args()

    print(f"args: {args}\n")

    if len(args.model) > 0 and len(args.dataset) > 0:
        main(args.model, args.dataset)
    elif len(args.dataset) > 0:
        for model_name in model_names:
            main(model_name, args.dataset)
    elif len(args.model) > 0:
        for dataset in datasets:
            main(args.model, dataset)
    else:
        for model_name in model_names:
            for dataset in datasets:
                main(model_name, dataset)
