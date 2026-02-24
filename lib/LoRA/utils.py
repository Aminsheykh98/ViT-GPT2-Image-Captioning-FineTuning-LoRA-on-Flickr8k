import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from typing import Optional, List

from .layers import LoRALinear, LoRAConv

def lora_trainable(model):
    """
    Freezes all model parameters except for those containing 'lora_' in their name.

    Args:
        model (nn.Module): The model to prepare for LoRA training.
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

def apply_lora(model, targets: List, r: int = 4, alpha: int = 1.0):
    """
    Recursively replaces targeted Linear or Conv1D layers with their LoRA counterparts.

    Args:
        model (nn.Module): The base model to transform.
        targets (list): List of strings containing layer names to match for replacement.
        r (int): The rank of the LoRA adaptation (default: 4).
        alpha (int): The scaling factor for the LoRA path (default: 1).

    Returns:
        nn.Module: The model with targeted layers wrapped in LoRA modules.
    """
    def replace_module(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            # Check if this is a target layer
            is_target = any(t in name for t in targets)
            
            if is_target:
                # HANDLE GPT-2 Conv1D
                if isinstance(child, Conv1D):
                    print(f"Applying LoRA to Conv1D: {full_name}")
                    lora_conv = LoRAConv(
                        conv1d=child,
                        r=r,
                        alpha=alpha
                    )
                    lora_conv.conv.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        lora_conv.conv.bias.data = child.bias.data.clone()
                    lora_conv.conv.weight.requires_grad = False
                    if lora_conv.conv.bias is not None:
                        lora_conv.conv.bias.requires_grad = False
                    setattr(module, name, lora_conv)
                
                # HANDLE Standard Linear (e.g., for ViT)
                elif isinstance(child, nn.Linear):
                    print(f"Applying LoRA to Linear: {full_name}")
                    lora_linear = LoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        r=r,
                        alpha=alpha
                    )
                    lora_linear.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        lora_linear.bias.data = child.bias.data.clone()
                    
                    # Freeze original weights in new layer
                    lora_linear.weight.requires_grad = False
                    if lora_linear.bias is not None:
                        lora_linear.bias.requires_grad = False
                    setattr(module, name, lora_linear)
                    
            
            else:
                # Recurse deeper
                replace_module(child, full_name)

    replace_module(model)
    return model

def print_fc_layers(model):
    """
    Recursively traverses the model to print the names and details of all Linear and Conv1D layers.

    Args:
        model (nn.Module): The model to inspect.
    """
    def replace_module(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            if isinstance(child, Conv1D) or isinstance(child, nn.Linear):
                print(f"Fully Connected Layer Name: {full_name, child}")
                    
            else:
                # Recurse deeper
                replace_module(child, full_name)
    replace_module(model)

def apply_merge(model):
    """
    Iterates through the model and triggers the weight merging process for all LoRA layers.

    Args:
        model (nn.Module): The model containing LoRA layers to be merged.

    Returns:
        nn.Module: The model with LoRA weights permanently fused into the base weights.
    """
    for name, module in model.named_modules():
        
        # 2. Get the name of the class as a string
        class_name = module.__class__.__name__
        
        # 3. Check if it's one of your LoRA types
        if class_name in ["LoRALinear", "LoRAConv"]:
            print(f"Merging layer: {name} (Type: {class_name})")
            
            # 4. Call the merge method
            # Note: Ensure your LoRALinear and LoRAConv both have the .merge() method
            module.merge()

    print("All LoRA layers have been merged into the base weights.")
    return model
