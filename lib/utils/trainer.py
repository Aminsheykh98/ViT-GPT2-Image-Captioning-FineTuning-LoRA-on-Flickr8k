import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore, ROUGEScore
import evaluate
import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import re

class Metrics:
    """
    Handles calculation, cleaning, and visualization of NLP evaluation metrics.

    Metrics include BLEU-1, BLEU-4, ROUGE-L, and METEOR.
    """
    def __init__(self):
        self.b1 = BLEUScore(n_gram=1)
        self.b4 = BLEUScore(n_gram=4)
        self.rouge = ROUGEScore()
        self.meteor = evaluate.load("meteor")

    def metrics(self, pred_text, target_text):
        """
        Calculates multiple NLP scores for a set of predicted texts against references.

        Args:
            pred_text (list): List of predicted strings.
            target_text (list): List of ground truth strings.

        Returns:
            tuple: (b1, b4, rouge, meteor) as float scores.
        """
        b1_score = self.b1(pred_text, target_text).item()
        b4_score = self.b4(pred_text, target_text).item()
        rouge_score = self.rouge(pred_text, target_text)['rougeL_fmeasure']
        meteor_score = self.meteor.compute(predictions=pred_text, references=target_text)["meteor"]
        return b1_score, b4_score, rouge_score, meteor_score

    @staticmethod
    def clean_metric(val):
        if isinstance(val, str):
            # Use regex to find the decimal number inside the string
            match = re.search(r"(\d+\.\d+)", val)
            return float(match.group(1)) if match else 0.0
        return val
    
    def plot_loss(self, df):
        """
        Generates and displays training loss and metric trends over epochs.

        Args:
            df (pd.DataFrame): DataFrame containing history of scores and training loss.
        """
        import matplotlib.pyplot as plt
        df['rouge'] = df['rouge'].apply(Metrics.clean_metric)
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
        axes = axes.flatten() # Flatten to 1D for easy looping
        metrics = ['loss_train', 'b1', 'b4', 'rouge', 'meteor']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']
        for i, metric in enumerate(metrics):
            epochs = range(1, len(df) + 1)
            axes[i].plot(epochs, df[metric], marker='o', color=colors[i], linewidth=2)
            # Formatting
            axes[i].set_title(f'{metric.upper()}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Score')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].set_xticks(epochs)
        # 4. Cleanup: Remove the empty 6th subplot
        fig.delaxes(axes[5])
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_scores(epoch, scores):
        if 'loss_train' not in scores.keys():
            scores['loss_train'] = 0.0
        print(f'Epoch: {epoch+1} | Train loss: {scores['loss_train']} | b1: {scores['b1']} | b4: {scores['b4']} | rouge: {scores['rouge']} | meteor: {scores['meteor']}')

class Trainer(Metrics):
    """
    Orchestrates the training, validation, and testing loops for the image captioning model.

    Args:
        num_epochs (int): Total training epochs.
        model (nn.Module): The vision-encoder-decoder model.
        optimizer (torch.optim): Optimization algorithm.
        tokenizer: Tokenizer for decoding/encoding text.
        train_loader, val_loader, test_loader: Data iterators.
        path_save (str): Directory to save weights and history.
        plot (bool): Whether to generate plots after training.
        device (torch.device): CPU or GPU hardware accelerator.
    """
    def __init__(
            self,
            num_epochs: int,
            model: nn.Module,
            optimizer,
            tokenizer,
            train_loader,
            val_loader,
            test_loader,
            path_save,
            plot,
            device
    ):
        super().__init__()
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        # model.config.dropout = 0.0              ## Overfit!
        # model.config.attention_dropout = 0.0    ## Overfit!
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device
        self.path_save = path_save
        self.plot = plot
        self.history = defaultdict(list)

    @staticmethod
    def unpack_batch(batch, device):
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        dec_mask = batch['decoder_attn_mask'].to(device)
        return images, labels, dec_mask

    def train(self):
        self.model.train(True)
        pbar = tqdm(self.train_loader, desc="Training")
        total_loss = 0
        for batch in pbar:
            images, labels, dec_mask = Trainer.unpack_batch(batch, self.device)
            self.optimizer.zero_grad()
            outputs = self.model(
                pixel_values=images,
                labels=labels, 
                decoder_attention_mask=dec_mask
            )
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def val(self):
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images = batch['images'].to(self.device)
                labels = batch['labels']

                image_attention_mask = torch.ones(images.shape[0], 197).to(self.device)
                generated_ids = self.model.generate(pixel_values=images,
                                                    attention_mask=image_attention_mask,
                                                    max_length=32,
                                                    early_stopping=True,
                                                    num_beams=4,
                                                    pad_token_id=self.tokenizer.pad_token_id, 
                                                    eos_token_id=self.tokenizer.eos_token_id)
                decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(labels)
            b1, b4, rouge, meteor = self.metrics(predictions, references)
        return {'b1': b1, 'b4': b4, 'rouge': rouge, 'meteor': meteor}
            
    
    def save_lora_weights(self):
        """
        Extracts and saves only the trainable (LoRA) parameters to a .pth file.
        """
        # 1. Filter the state dict for only trainable parameters
        lora_state_dict = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        path_save_lora = os.path.join(self.path_save, 'lora_weights.pth')
        torch.save(lora_state_dict, path_save_lora)
        print(f"Trainable LoRA weights saved to {path_save_lora}")

    def save_history(self, scores=None, to_csv=False):
        """
        Logs epoch scores to internal history or exports them to a CSV file.

        Args:
            scores (dict, optional): Current epoch scores to log.
            to_csv (bool): If True, saves history to disk and generates plots.
        """
        if to_csv:
            df = pd.DataFrame(self.history)
            df.to_csv(os.path.join(self.path_save, 'history.csv'), index=False)
            if self.plot:
                self.plot_loss(df)

        else:
            for metric, score in scores.items():
                self.history[metric].append(score)
    
    def test(self):
        path_save = os.path.join(self.path_save, 'lora_weights.pth')
        lora_weights = torch.load(path_save) 
        self.model.load_state_dict(lora_weights, strict=False)
        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                images = batch['images'].to(self.device)
                labels = batch['labels']

                image_attention_mask = torch.ones(images.shape[0], 197).to(self.device)
                generated_ids = self.model.generate(pixel_values=images,
                                                    attention_mask=image_attention_mask,
                                                    max_length=32,
                                                    early_stopping=True,
                                                    num_beams=4,
                                                    pad_token_id=self.tokenizer.pad_token_id, # Last two carefully!
                                                    eos_token_id=self.tokenizer.eos_token_id)
                decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(labels)
            b1, b4, rouge, meteor = self.metrics(predictions, references)
            print(f'*Test Scores* | b1: {b1} | b4: {b4} | rouge: {rouge} | meteor: {meteor}')

    def __call__(self):
        scores_init = self.val()
        Trainer.print_scores(-1, scores_init)
        rouge_best = scores_init['rouge']
        for epoch in range(self.num_epochs):
            loss_train = self.train()
            scores = self.val()
            scores['loss_train'] = loss_train
            self.save_history(scores)
            Trainer.print_scores(epoch, scores)
            if scores['rouge'] > rouge_best:
                self.save_lora_weights()
                rouge_best = scores['rouge']

        self.save_history(to_csv=True)
        self.test()