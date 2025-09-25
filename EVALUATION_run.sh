CUDA_VISIBLE_DEVICES=3 py EVALUATION_country.py
notify "country"
CUDA_VISIBLE_DEVICES=3 py EVALUATION_lskextractor-top3.py
notify "top3"
CUDA_VISIBLE_DEVICES=3 py EVALUATION_lskextractor.py
notify "lsk"
CUDA_VISIBLE_DEVICES=3 py EVALUATION_majority.py
notify "majoriy"
CUDA_VISIBLE_DEVICES=3 py EVALUATION_one_language.py
notify "one lang"
CUDA_VISIBLE_DEVICES=3 py EVALUATION_only_english.py
notify "english"
# CUDA_VISIBLE_DEVICES=2,3 py EVALUATION_llm_selected.py