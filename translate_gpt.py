# from vllm import LLM, SamplingParams
# import torch
import json
import pickle
from tqdm import tqdm

# import openai
# from parallel_inference import InferenceEngine, Hyperparameters

DATASET = "culture_atlas"  # "blend", "social_iqa", "culture_atlas"

with open(f'data/{DATASET}/data.pkl', 'rb') as f:
    data = pickle.load(f)

data = data.sample(n=min(50000, len(data)), random_state=512)

data_input = data['input']
data_output = data['output']
data_choices = data['choices']

set_inputs = list(set(data_input))
set_choices = set()
for choice in data_choices:
    temp = choice.replace('A.', '').replace('B.', '').replace('C.', '').replace('D.', '').split('\n')
    for c in temp:
        set_choices.add(c.strip())

set_choices = list(set_choices)

#### SET UP GPT ####
import os
from openai import AzureOpenAI

endpoint = os.getenv("AZURE_ENDPOINT")
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)
#### SET UP GPT ####

def translate(list_input, language):
    if "English" in language:
        return list_input
    translation_format = lambda language, input: f"""Translate ONLY the following question into {language}: \"{input}\". ONLY output the translation in the following JSON format:

{{
    \"{language}_translation\": <output the translated input here>.
}}

"""

    input_queries = [translation_format(language, iq) for iq in list_input]
    translated_responses = []
    for input_query in tqdm(input_queries, desc=f"translating into {language}"):
        try:
            out = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": input_query,
                    }
                ],
                max_completion_tokens=800,
                temperature=1.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                model=deployment
            ).choices[0].message.content

            translated_responses.append(json.loads(out[out.index("{"):])[f'{language}_translation'])
        except:
            translated_responses.append(None)
    return translated_responses


from config import LANGUAGE_SET

for LANG in LANGUAGE_SET:
    if os.path.exists(f'data/{DATASET}/data_{LANG}.pkl'):
        print(f"data/{DATASET}/data_{LANG}.pkl exists, skipping...")
        continue
    translated_inputs = translate(set_inputs, LANG)
    translated_choices = translate(set_choices, LANG)

    inputs, choices, outputs = [None] * len(data), [None] * len(data), [None] * len(data)
    count = 0
    for i, c, o in zip(data_input, data_choices, data_output):
        inputs[count] = (translated_inputs[set_inputs.index(i)])

        temp = [x[3:] for x in c.split('\n')]
        tc = []
        for t in temp:
            tc.append(translated_choices[set_choices.index(t.strip())])
        
        if DATASET == "blend":
            choices[count] = (f"A. {tc[0]}\nB. {tc[1]}\nC. {tc[2]}\nD. {tc[3]}")
        if DATASET == "social_iqa":
            choices[count] = (f"A. {tc[0]}\nB. {tc[1]}\nC. {tc[2]}")
        if DATASET == "culture_atlas":
            choices[count] = (f"A. {tc[0]}\nB. {tc[1]}")

        outputs[count] = (f"{o[:3]}{translated_choices[set_choices.index(o[3:].strip())]}")

        count += 1

    
    data['input'] = inputs
    data['choices'] = choices
    data['output'] = outputs

    with open(f'data/{DATASET}/data_{LANG}.pkl', 'wb+') as f:
        pickle.dump(data, f)
