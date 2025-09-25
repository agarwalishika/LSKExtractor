from tqdm import tqdm
import pandas as pd
import pickle
import json

# https://drive.google.com/drive/folders/1ueUjBCqCxH2ZUAHmG1qGJ6XcC7or8fMZ
with open('culture_info_beta_Nov21-001.json', 'r') as f:
    json_data = json.load(f)

countries, topics, subtopics, subsubtopics, questions, answers, choices = [], [], [], [], [], [], []
for country in tqdm(json_data.keys()):
    if country == "metadata":
        continue
    for topic in json_data[country].keys():         
        for subtopic in json_data[country][topic].keys():
            for subsubtopic in json_data[country][topic][subtopic].keys():
                if 'norm_violation_rlv' not in json_data[country][topic][subtopic][subsubtopic].keys():
                    continue
                for sentence in json_data[country][topic][subtopic][subsubtopic]['norm_violation_rlv']:
                    countries.append(country)
                    topics.append(topic)
                    subtopics.append(subtopic)
                    subsubtopics.append(subsubtopic)

                    if "Steinbach, Manitoba" in subsubtopic:
                        hi = 9
                    questions.append(sentence)
                    if json_data[country][topic][subtopic][subsubtopic]['norm_violation_rlv'][sentence][0] == "Yes":
                        answers.append("A. True")
                    else:
                        answers.append("B. False")
                    choices.append("A. True\nB. False")

df = pd.DataFrame.from_dict({"country": countries, "topic": topics, "subtopic": subtopics, "subsubtopic": subsubtopics, "input": questions, "output": answers, "choices": choices})

with open('data.pkl', 'wb+') as f:
    pickle.dump(df, f)