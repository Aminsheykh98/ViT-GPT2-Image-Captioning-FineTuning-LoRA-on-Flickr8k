import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore, ROUGEScore
import evaluate

class Metrics:
    def __init__(self):
        self.b1 = BLEUScore(n_gram=1)
        self.b4 = BLEUScore(n_gram=4)
        self.rouge = ROUGEScore()
        self.meteor = evaluate.load("meteor")

    def metrics(self, pred_text, target_text):
        b1_score = self.b1(pred_text, target_text).item()
        b4_score = self.b4(pred_text, target_text).item()
        rouge_score = self.rouge(pred_text, target_text)['rougeL_fmeasure']
        meteor_score = self.meteor.compute(predictions=pred_text, references=target_text)["meteor"]
        return b1_score, b4_score, rouge_score, meteor_score
    
class Tester(Metrics):
    def __init__(
            self,
            model: nn.Module,
            tokenizer,
            test_loader,
            device
    ):
        super().__init__()
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device


    def val(self):
        self.model.eval()
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
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    eos_token_id=self.tokenizer.eos_token_id)
                decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(labels)
            b1, b4, rouge, meteor = self.metrics(predictions, references)
        return {'b1': b1, 'b4': b4, 'rouge': rouge, 'meteor': meteor}
            
    def __call__(self):
        scores = self.val()
        print(f'*Test Scores* | b1: {scores['b1']} | b4: {scores['b4']} | rouge: {scores['rouge']} | meteor: {scores['meteor']}')

