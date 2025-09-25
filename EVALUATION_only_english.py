
from config import *
from glob import glob
import pandas as pd

def main(model_name, dataset, suffix):
    data_files = glob(f"generations_json/{dataset}*{model_name}{suffix}.json")
    data_file = [f for f in data_files if "English" in f][0]

    df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
    num_accurate = sum(df['final_parse_answer'] == df['output'])

    accuracy = num_accurate / NUM_TEST
    with open('results_OnlyEnglish.txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{accuracy}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{accuracy}\n")
        

    
for qa_dataset in EVALUATION_DATASETS:
    for llm in EVALUATION_LLMS:
        for suffix in ["", "_NR"]:
            print(f"Model: {llm}, Dataset: {qa_dataset}")
            try:
                main(llm.replace('/', '_'), qa_dataset, suffix)
            except:
                continue