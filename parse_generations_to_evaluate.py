import os
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

import json
import pickle
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
import os

import numpy as np

def semantic_sim(responses, outputs):
    """
    Batch process semantic similarity between responses and their corresponding choices.
    Uses batched embedding calls for efficiency.
    
    Args:
        responses: List of response strings
        choices_list: List of choice strings (each will be split by '\n')
    
    Returns:
        List of final answers (one for each response)
    """
    # Batch encode all responses at once
    response_embs = embedding_model.encode([response.strip() for response in responses])
    output_embs = embedding_model.encode([output.strip() for output in outputs])
    
    similarities = []
    for r, o in zip(response_embs, output_embs):
        similarities.append(np.dot(o, r) / (np.linalg.norm(r) * np.linalg.norm(o)).item())
    
    return similarities

from config import *
files = glob('generations_colm/*/*.pkl')
os.makedirs('generations_parsed/', exist_ok=True)

for file in files:
    f = file.split('/')[-1]
    output_file = "generations_parsed/" + f[:f.rfind('.')] + ".json"

    if os.path.exists(output_file):
        print(f'{output_file} exists, skip.')
        continue

    with open(file, 'rb') as f:
        data = pickle.load(f)
    data = data[-NUM_TEST:]

    responses = list(data.apply(lambda row: row['vllm_response'].strip(), axis=1))
    outputs = list(data['output'])
    
    st_final_answers = semantic_sim(responses, outputs)
    
    data['final_parse_answer'] = st_final_answers
    data['is_correct'] = [1.0 if sim > 0.7 else 0.0 for sim in st_final_answers]
    data.to_json(output_file)