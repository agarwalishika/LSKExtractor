import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

def semantic_sim(responses, choices_list):
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
    print(f'processing {len(responses)} responses...')
    response_embs = embedding_model.encode([response.strip() for response in responses])
    
    # Split all choices and flatten for batch encoding
    all_choices = []
    choice_indices = []  # Track which choices belong to which question
    
    for i, choices in enumerate(choices_list):
        choices_split = choices.split('\n')
        all_choices.extend(choices_split)
        choice_indices.extend([i] * len(choices_split))
    
    # Batch encode all choices at once
    all_choices_embs = embedding_model.encode(all_choices)
    
    # Process each response-choices pair
    final_answers = []
    choice_start_idx = 0
    
    for i, choices in enumerate(choices_list):
        if i == 26: 
            hi = 9
        choices_split = choices.split('\n')
        num_choices = len(choices_split)
        
        # Get embeddings for this question's choices
        choices_embs = all_choices_embs[choice_start_idx:choice_start_idx + num_choices]
        response_emb = response_embs[i]
        
        # Calculate cosine similarity
        cosine_similarities = np.dot(choices_embs, response_emb) / (np.linalg.norm(response_emb) * np.linalg.norm(choices_embs, axis=1))
        final_answer = choices_split[np.argmax(cosine_similarities)]
        final_answers.append(final_answer)
        
        choice_start_idx += num_choices
    
    return final_answers

def extract_value(response, choices):
    temp = [c[2:] for c in choices.split('\n')]
    count, final_answer = 0, ''
    for i, t in enumerate(temp):
        if t.strip().lower() in response.strip().lower():
            count += 1
            final_answer = choices.split('\n')[i]

    if count != 1:
        return None
    return final_answer

def extract_number(response, choices):
    temp = [c[0] for c in choices.split('\n')]
    count, final_answer = 0, ''
    for i, t in enumerate(temp):
        if t.strip() in response.strip():
            count += 1
            final_answer = choices.split('\n')[i]

    if count != 1:
        return extract_value(response, choices)
    return final_answer

def extract(response, choices):
    temp = choices.split('\n')
    count, final_answer = 0, ''
    for t in temp:
        if t in response:
            count += 1
            final_answer = t

    if count != 1:
        return extract_number(response, choices)
    return final_answer

import re
def smart_split(text):
    """
    Split a string formatted like 'A. text... B. text... C. text...' 
    into a list of strings, ignoring newlines inside each section.
    """
    # Match "A.", "B.", "C.", "D." followed by text until next letter-dot or end
    parts = re.split(r'\n?(?:[A-D]\.)', text)
    
    # First split will give an empty string before "A.", so remove empties
    parts = [p.strip().replace("\n", " ") for p in parts if p.strip()]
    
    return f"A. {parts[0]}\nB. {parts[1]}\nC. {parts[2]}\nD. {parts[3]}"

from config import *
files = glob('generations/blend*.pkl')
files.extend(glob('generations/culture*.pkl'))
files.extend(glob('generations/social*.pkl'))
for file in files:
    f = file.split('/')[-1]
    output_file = "generations_json/" + f[:f.rfind('.')] + ".json"

    if os.path.exists(output_file):
        print(f'{output_file} exists, skip.')
        continue

    with open(file, 'rb') as f:
        data = pickle.load(f)
    data = data[-NUM_TEST:]

    responses = list(data.apply(lambda row: row['vllm_response'].strip(), axis=1))
    choices = list(data['choices'])
    final_answers = []
    
    for r, c in tqdm(zip(responses, choices), total=len(responses)):
        if len(c.split('\n')) > 4:
            c = smart_split(c)
        temp = extract(r, c)
        if temp is None and "final_answer" in r:
            temp = extract(r[r.find('final_answer'):], c)
        if temp is None and "final_answer" in r:
            temp = extract(r[r.find('final_answer'):].strip().split('\n')[0], c)
        if temp is None:
            hi = 9
            extract(r[r.find('final_answer'):], c)
            extract(r[r.find('final_answer'):].strip().split('\n')[0], c)
        final_answers.append(temp)
    
    st_responses = []
    st_choices = []
    for i, fas in enumerate(final_answers):
        if fas is None:
            if len(choices[i].split('\n')) > 4:
                c = smart_split(choices[i])
            else:
                c = choices[i]
            st_responses.append(responses[i])
            st_choices.append(c)
    
    st_final_answers = semantic_sim(st_responses, st_choices)
    counter = 0
    for i, fas in enumerate(final_answers):
        if fas is None:
            final_answers[i] = st_final_answers[counter]
            counter += 1
    
    data['final_parse_answer'] = final_answers
    data.to_json(output_file)