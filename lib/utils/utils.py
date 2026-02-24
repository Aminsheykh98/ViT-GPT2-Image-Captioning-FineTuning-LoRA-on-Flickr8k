import torch
import os
import json

def print_num_params(model):
    """
    Calculates and prints the count of total, trainable, and frozen parameters in a model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")

def save_hyperparameters(path, hyperparams):
    """
    Saves a dictionary of hyperparameters to a JSON file in the specified directory.

    Args:
        path (str): The directory where the JSON file will be saved.
        hyperparams (dict): A dictionary containing hyperparameter names and values.
    """
    file_path = os.path.join(path, "hyperparameters.json")
    with open(file_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f'Hyperparameters saved to: {file_path}')

def return_collate_fn(tokenizer):
    """
    A closure that returns a collate function tailored for training with a specific tokenizer.

    Args:
        tokenizer: The tokenizer used to process text captions.

    Returns:
        callable: A collate_fn that tokenizes captions, handles EOS tokens, and masks padding for loss calculation.
    """
    def collate_fn(batch):
        # 1. Extract images and captions
        # Assuming batch is [(img, caption), ...]
        # images = [item[0] for item in batch]
        images, caption_lists = zip(*batch)
        images = [img['pixel_values'][0] for img in images]
        captions = [cap for cap in caption_lists]
        
        # 2. Add EOS token to the end of every caption manually!
        # This ensures the model learns where the sentence STOPS.
        captions = [c[0] + tokenizer.eos_token for c in captions]

        # 3. Tokenize
        tokenized = tokenizer(
            captions,
            padding="max_length", 
            truncation=True,
            max_length=32, # Ensure this matches your val max_length
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        
        # 4. Create Labels (The Critical Fix)
        labels = input_ids.clone()
        
        # Wherever attention_mask is 0 (padding), set label to -100
        # This tells PyTorch: "Don't calculate loss for these tokens"
        labels[attention_mask == 0] = -100
        
        # 5. Process Images
        # (Assuming you have a transforms function applied already or here)
        # If images are already tensors:
        pixel_values = torch.stack(images)
        
        return {
            "images": pixel_values,
            "input_ids": input_ids,
            "decoder_attn_mask": attention_mask,
            "labels": labels
        }
    return collate_fn

def collate_fn_eval(batch):
    """
    Processes a batch for evaluation, keeping raw captions as strings for metric calculation.

    Args:
        batch (list): A list of tuples containing (image, caption_list).

    Returns:
        dict: A dictionary containing stacked image tensors and raw label lists.
    """
    images, caption_lists = zip(*batch)
    images = [img['pixel_values'][0] for img in images]
    return {
            "images": torch.stack(images),
            "labels": list(caption_lists),
            "decoder_attn_mask": None
        }