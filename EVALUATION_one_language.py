from config import *
from glob import glob
import pandas as pd

def main(model_name, dataset, suffix):
    data_files = glob(f"generations_parsed/{dataset}*{model_name}{suffix}.json")
    assert len(data_files) == 15

    best_acc, best_lang = 0, ""
    for data_file in data_files:
        df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
        acc = sum(df['is_correct']) / NUM_TEST
        
        if acc > best_acc:
            best_acc = acc
            best_lang = pick_out_lang(data_file)

    with open('results_GlobalLanguage.txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{best_acc}\t{best_lang}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{best_acc}\t{best_lang}\n")
        

    
for qa_dataset in EVALUATION_DATASETS:
    for llm in EVALUATION_LLMS:
        # for suffix in ["", "_NR"]:
        for suffix in [""]:
            print(f"Model: {llm}, Dataset: {qa_dataset}")
            try:
                main(llm.replace('/', '_'), qa_dataset, suffix)
            except:
                continue
