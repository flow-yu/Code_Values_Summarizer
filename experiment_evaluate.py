from transformers import RobertaTokenizer, T5ForConditionalGeneration
from utile import *
import json
import re
import os
import subprocess
import time
import sys
from tqdm import tqdm

def evaluate_code_sum(datasets, code_only = True, model = "model_checkpoints/code_only_epoch2", label_path = "Starcoder_sum/starcoder_tags.json"):
    print(f"Evaluating the model {model}")
    tokenizer = RobertaTokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model)

    label_dict = read_json_file(label_path)
    for dataset in datasets:
        print (f"Evaluating {dataset}")
        dataset_dict = read_json_file(dataset)
        predictions = []
        labels = []
        
        # Take half of the dataset for testing
        dataset_dict = dict(list(dataset_dict.items())[:len(dataset_dict)//2])
        pbar = tqdm(
            total=len(dataset_dict), 
            desc="Processing", 
            unit="example", 
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}{r_bar}"
        )

        for id, problem in dataset_dict.items():
            if code_only:
                text = problem['code_only']
            else:
                text = problem['code_values']
            input_ids = tokenizer(text, return_tensors="pt").input_ids   
            generated_ids = model.generate(input_ids)
            output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(output)
            labels.append(label_dict[id])

            pbar.update(1)

        # Close the progress bar after finishing the loop
        pbar.close()

        score_metric = compute_bleu_2gram(predictions, labels)
        print (f"BLEU score for {dataset} is {score_metric}")

# datasets = ["Starcoder_sum/code_sum_chunk1.json", "Starcoder_sum/code_sum_chunk2.json", "Starcoder_sum/code_sum_chunk3.json", "Starcoder_sum/code_sum_chunk4.json"]
datasets = ["Starcoder_sum/code_sum_chunk2.json"]
evaluate_code_sum(datasets)
