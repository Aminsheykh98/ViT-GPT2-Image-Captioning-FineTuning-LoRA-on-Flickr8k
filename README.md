# ViT-GPT2 Image Captioning on Flickr8k with LoRA

This repository contains an implementation of image captioning using a ViT-GPT2 architecture, fine-tuned on the Flickr8k dataset.
The project focuses on parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) without relying on external PEFT libraries.

## Project Overview

Image captioning is a multi-modal task that maps visual inputs to natural language descriptions. In this project:

  * Vision Transformer (ViT) is used as the image encoder
  * GPT-2 is used as the text decoder
  * LoRA is applied to selectively fine-tune attention layers while keeping the majority of the model frozen
  * Training and evaluation are performed on Flickr8k, which provides 5 captions per image
  * The project investigates:
  * Captioning performance using BLEU, METEOR, and ROUGE
  * The effect of LoRA rank on model stability and generalization
  * Internal loss behavior of the VisionEncoderDecoder model

## Model Architecture

Image → ViT Encoder → Cross-Attention → GPT-2 Decoder → Caption

Encoder: ViT-base (patch-based image representation)
Decoder: GPT-2 with cross-attention

Fine-tuning: LoRA applied to selected attention layers (default: query, value, c_attn. Can be changed by setting up the right layer's names.)

## Dataset
The dataset can be found in:

https://github.com/goodwillyoga/Flickr8k_dataset?tab=readme-ov-file

And put the unzipped dataset file in this repository.

