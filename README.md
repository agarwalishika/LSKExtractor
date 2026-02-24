# Running LSK

First we need to get all of the model's outputs in all various languages. Once we have that, we can form inference-time as a "language selection" problem. To do so, we simply run `run_inference.py`. If you want to run without reasoning, you run `run_inference_nr.py`.
- `run_inference.py` will perform inference with all your models, all your languages, and all your datasets. It will ask for the model's answer in all settings, and save them to a folder called `generations`.

Next, to make evaluation easier, we need to parse the model responses into answer choices. Hence, run `parse_generations_to_classify.py`. Now, we can formulate this as a language selection problem.

Finally, evaluate the language selection using all the `EVALUATION_*.py` files.