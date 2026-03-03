from config import *
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import mode

def main(model_name, dataset, suffix):
    data_files = glob(f"generations_parsed/{dataset}*{model_name}{suffix}.json")
    try:
        assert len(data_files) == 15
    except:
        import pdb; pdb.set_trace()

    voted_answers = []
    for data_file in data_files:
        print(data_file)
        df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
        voted_answers.append(list(df.apply(lambda x: x['is_correct'], axis=1)))
        
    num_accurate = np.array(voted_answers).mean()

    accuracy = num_accurate / NUM_TEST
    with open('results_Majority.txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{accuracy}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{accuracy}\n")
        

    
for qa_dataset in EVALUATION_DATASETS:
    for llm in EVALUATION_LLMS:
        for suffix in [""]:
        # for suffix in ["", "_NR"]:
            # try:
                print(f"Model: {llm}, Dataset: {qa_dataset}")
                main(llm.replace('/', '_'), qa_dataset, suffix)
            # except:
            #     continue
