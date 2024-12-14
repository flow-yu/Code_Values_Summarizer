import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import RobertaTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from utile import *

def train_code_sum(datasets, 
                   code_only=True, 
                   label_path="Starcoder_sum/starcoder_tags.json",
                   model_name="codet5-base-multi-sum",
                   epochs=2, 
                   learning_rate=5e-5, 
                   gradient_accumulation_steps=1,
                   batch_size=32,  
                   save_dir="model_checkpoints"):

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    label_dict = read_json_file(label_path)

    # Gather all (dataset_file, id) pairs
    all_ids = []
    for dataset in datasets:
        dataset_dict = read_json_file(dataset)
        for ex_id in dataset_dict.keys():
            if ex_id in label_dict:
                all_ids.append((dataset, ex_id))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(all_ids) * epochs // gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        
        # Shuffle data for each epoch
        indices = torch.randperm(len(all_ids))

        pbar = tqdm(
            total=len(all_ids),
            desc=f"Epoch {epoch+1}",
            unit="example", 
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}{r_bar}"
        )
        running_loss = 0.0
        optimizer.zero_grad()

        # Temporary storage for a batch
        batch_inputs = []
        batch_labels = []
        
        for step, idx in enumerate(indices):
            dataset_file, ex_id = all_ids[idx]
            dataset_dict = read_json_file(dataset_file)
            problem = dataset_dict[ex_id]

            if code_only:
                text = problem['code_only']
            else:
                text = problem['code_values']
            target_text = label_dict[ex_id]

            batch_inputs.append(text)
            batch_labels.append(target_text)

            # If we have enough samples to form a batch or we are at the end
            if len(batch_inputs) == batch_size or (step == len(all_ids)-1 and len(batch_inputs) > 0):

                # Tokenize in batch
                input_encodings = tokenizer(batch_inputs, return_tensors="pt", 
                                            padding=True, truncation=True, max_length= 512)
                label_encodings = tokenizer(batch_labels, return_tensors="pt", 
                                            padding=True, truncation=True, max_length=40)
                
                input_ids = input_encodings.input_ids.to(device)
                attention_mask = input_encodings.attention_mask.to(device)
                labels = label_encodings.input_ids.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                running_loss += loss.item()

                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Reset batch
                batch_inputs = []
                batch_labels = []

            pbar.update(1)
            pbar.set_postfix({"loss": f"{running_loss/(step+1):.4f}"})

        pbar.close()
        print(f"Epoch {epoch+1} finished. Average loss: {running_loss/len(all_ids):.4f}")
        
        # Save model checkpoint after each epoch
        save_path = f"{save_dir}/epoch_{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    print("Training completed.")

train_code_sum(["Starcoder_sum/code_sum_chunk1.json", "Starcoder_sum/code_sum_chunk3.json", "Starcoder_sum/code_sum_chunk4.json"])