# step 0: translate all inputs
py translate_inputs.py

# step 1a: llm inference
py main_translate.py
# step 1b: inference evaluation
py main_evaluate_translate.py
# step 1c: cluster and label
py cluster.py

# visualize results so far
mkdir figures_difference
mkdir figures_only_with
mkdir figures_percent_change
py visualize_translate.py

# step 2: use mlk knowledge in inference
py main_test.py