from config import *
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import mode

def main(model_name, dataset, suffix):
    data_files = glob(f"generations_json/{dataset}*{model_name}{suffix}.json")
    assert len(data_files) == 16

    voted_answers = []
    for data_file in data_files:
        print(data_file)
        df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
        voted_answers.append(list(df.apply(lambda x: let_to_num[x['final_parse_answer'].split('.')[0]], axis=1)))
    
    majority_answers = mode(np.array(voted_answers), axis=0).mode
    output_numerical_answers = list(df.apply(lambda x: let_to_num[x['output'].split('.')[0]], axis=1))
    num_accurate = sum(majority_answers == output_numerical_answers)

    accuracy = num_accurate / NUM_TEST
    with open('results_Majority.txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{accuracy}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{accuracy}\n")
        

    
for qa_dataset in EVALUATION_DATASETS:
    for llm in EVALUATION_LLMS:
        for suffix in ["", "_NR"]:
            # try:
                print(f"Model: {llm}, Dataset: {qa_dataset}")
                main(llm.replace('/', '_'), qa_dataset, suffix)
            # except:
            #     continue