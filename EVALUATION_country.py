import os
from config import *
from glob import glob
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from config import *
import pickle
from glob import glob
import torch

blend_country_map = {
    'Algeria': "Arabic",
    'Assam': "English",
    'Azerbaijan': "English",
    'China': "Chinese",
    'Ethiopia': "Arabic",
    'Greece': "English",
    'Indonesia': "English",
    'Iran': "English",
    'Mexico': "Spanish",
    'North_Korea': "Korean",
    'Northern_Nigeria': "English",
    'South_Korea': "Korean",
    'Spain': "Spanish",
    'UK': "English",
    'US': "English",
    'West_Java': "English"
}

culture_atlas_country_map = {
 'Afghanistan': "English",
 'Albania': "English",
 'Algeria': "Arabic",
 'Andorra': "English",
 'Angola': "Portuguese",
 'Antigua and Barbuda': "English",
 'Argentina': "Spanish",
 'Armenia': "English",
 'Australia': "English",
 'Austria': "German",
 'Azerbaijan': "English",
 'Bahamas': "English",
 'Bahrain': "Arabic",
 'Bangladesh': "Bengali",
 'Barbados': "English",
 'Belarus': "English",
 'Belgium': "English",
 'Belize': "English",
 'Benin': "French",
 'Bhutan': "English",
 'Bolivarian Republic of Venezuela': "Spanish",
 'Bosnia and Herzegovina': "English",
 'Botswana': "English",
 'Brazil': "Portuguese",
 'Bulgaria': "English",
 'Burkina Faso': "French",
 'Burundi': "English",
 'Cambodia': "English",
 'Cameroon': "French",
 'Canada': "English",
 'Central African Republic': "French",
 'Chad': "French",
 'Chile': "Spanish",
 'China': "Chinese",
 'Colombia': "Spanish",
 'Comoros': "Arabic",
 'Congo': "French",
 'Costa Rica': "Spanish",
 'Croatia': "English",
 'Cuba': "Spanish",
 'Cyprus': "English",
 'Czechia': "English",
 "Côte d'Ivoire": "French",
 "Democratic People's Republic of Korea": "Korean",
 'Democratic Republic of the Congo': "French",
 'Denmark': "English",
 'Djibouti': "French",
 'Dominica': "English",
 'Dominican Republic': "Spanish",
 'Ecuador': "Spanish",
 'Egypt': "Arabic",
 'El Salvador': "Spanish",
 'Equatorial Guinea': "Spanish",
 'Eritrea': "English",
 'Estonia': "English",
 'Eswatini': "English",
 'Ethiopia': "English",
 'Federated States of Micronesia': "English",
 'Fiji': "English",
 'Finland': "English",
 'France': "French",
 'Gabon': "French",
 'Gambia': "English",
 'Georgia': "English",
 'Germany': "German",
 'Ghana': "English",
 'Greece': "English",
 'Grenada': "English",
 'Guatemala': "Spanish",
 'Guinea': "French",
 'Guinea-Bissau': "Portuguese",
 'Guyana': "English",
 'Haiti': "English",
 'Honduras': "Spanish",
 'Hungary': "English",
 'Iceland': "English",
 'India': "Hindi",
 'Indonesia': "English",
 'Iraq': "Arabic",
 'Ireland': "English",
 'Islamic Republic of Iran': "English",
 'Israel': "English",
 'Italy': "Italian",
 'Jamaica': "English",
 'Japan': "Japanese",
 'Jordan': "Arabic",
 'Kazakhstan': "English",
 'Kenya': "English",
 'Kiribati': "English",
 'Kuwait': "Arabic",
 'Kyrgyzstan': "English",
 "Lao People's Democratic Republic": "English",
 'Latvia': "English",
 'Lebanon': "Arabic",
 'Lesotho': "English",
 'Liberia': "English",
 'Libya': "Arabic",
 'Liechtenstein': "German",
 'Lithuania': "English",
 'Luxembourg': "English",
 'Madagascar': "English",
 'Malawi': "English",
 'Malaysia': "English",
 'Maldives': "English",
 'Mali': "French",
 'Malta': "English",
 'Marshall Islands': "English",
 'Mauritania': "Arabic",
 'Mauritius': "English",
 'Mexico': "Spanish",
 'Monaco': "French",
 'Mongolia': "English",
 'Montenegro': "English",
 'Morocco': "Arabic",
 'Mozambique': "Portuguese",
 'Myanmar': "English",
 'Namibia': "English",
 'Nauru': "English",
 'Nepal': "English",
 'Netherlands': "English",
 'New Zealand': "English",
 'Nicaragua': "Spanish",
 'Niger': "French",
 'Nigeria': "English",
 'North Macedonia': "English",
 'Norway': "English",
 'Oman': "Arabic",
 'Pakistan': "English",
 'Palau': "English",
 'Panama': "Spanish",
 'Papua New Guinea': "English",
 'Paraguay': "Spanish",
 'Peru': "Spanish",
 'Philippines': "English",
 'Plurinational State of Bolivia': "Spanish",
 'Poland': "English",
 'Portugal': "Portuguese",
 'Qatar': "Arabic",
 'Republic of Korea': "Korean",
 'Republic of Moldova': "English",
 'Romania': "English",
 'Russian Federation': "Russian",
 'Rwanda': "English",
 'Saint Kitts and Nevis': "English",
 'Saint Lucia': "English",
 'Saint Vincent and the Grenadines': "English",
 'Samoa': "English",
 'San Marino': "Italian",
 'Saudi Arabia': "Arabic",
 'Senegal': "French",
 'Serbia': "English",
 'Seychelles': "English",
 'Sierra Leone': "English",
 'Singapore': "English",
 'Slovakia': "English",
 'Slovenia': "English",
 'Solomon Islands': "English",
 'Somalia': "English",
 'South Africa': "English",
 'South Sudan': "English",
 'Spain': "Spanish",
 'Sri Lanka': "English",
 'Sudan': "Arabic",
 'Suriname': "English",
 'Sweden': "English",
 'Switzerland': "German",
 'São Tomé and Príncipe': "Portuguese",
 'Tajikistan': "English",
 'Thailand': "Thai",
 'Timor-Leste': "English",
 'Togo': "French",
 'Tonga': "English",
 'Trinidad and Tobago': "English",
 'Tunisia': "Arabic",
 'Turkmenistan': "English",
 'Tuvalu': "English",
 'Türkiye': "Turkish",
 'Uganda': "English",
 'Ukraine': "English",
 'United Arab Emirates': "Arabic",
 'United Kingdom of Great Britain and Northern Ireland': "English",
 'United Republic of Tanzania': "English",
 'United States of America': "English",
 'Uruguay': "Spanish",
 'Uzbekistan': "English",
 'Vanuatu': "English",
 'Viet Nam': "Vietnamese",
 'Yemen': "Arabic",
 'Zambia': "English",
 'Zimbabwe': "English"
}




def main(model_name, dataset, suffix, country_map):
    data_files = glob(f"generations_json/{dataset}*{model_name}{suffix}.json")
    assert len(data_files) == 16

    num_acccurate = 0
    num_total = 0
    for data_file in data_files:
        df = pd.read_json(open(data_file, 'r'))[-NUM_TEST:]
        df_lang = pick_out_lang(data_file)

        for k in country_map.keys():
            if country_map[k] != df_lang:
                continue
            df_country = df[df['country'] == k]
            num_acccurate += sum(df_country['final_parse_answer'] == df_country['output'])
            num_total += len(df_country)
            break

    with open('results_Country.txt', 'a+') as f:
        if "NR" in data_file:
            f.write(f"NR,{model_name}\t{dataset}\t{num_acccurate / num_total}\n")
        else:
            f.write(f"WR,{model_name}\t{dataset}\t{num_acccurate / num_total}\n")
    
country_map = None
for qa_dataset in EVALUATION_DATASETS:
    if "social" in qa_dataset: continue
    if "blend" in qa_dataset: country_map = blend_country_map
    if "culture" in qa_dataset: country_map = culture_atlas_country_map
    for llm in EVALUATION_LLMS:
        for suffix in ["", "_NR"]:
            print(f"Model: {llm}, Dataset: {qa_dataset}")
            main(llm.replace('/', '_'), qa_dataset, suffix, country_map)
