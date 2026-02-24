import torch
from torch.utils.data import DataLoader
import os
from functools import partial
from time import gmtime, strftime
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from lib.data import load_data
from lib.LoRA.utils import apply_lora
import lib.utils.utils as u
from lib.utils.trainer import Trainer


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning",
                                                  local_files_only=True)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning",
                                                      local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning",
                                          local_files_only=True)

tokenizer.pad_token = tokenizer.eos_token

feature_extractor = partial(feature_extractor, return_tensors="pt")

dir_Flickr_jpg = 'dataset\\Flickr8k_Dataset'
dir_Flickr_text = 'dataset\\Flickr8k_Text\\Flickr8k.token.txt'
src_id_root = 'dataset\\Flickr8k_Text\\Flickr_8k.'
src_id = {'train': src_id_root + 'trainImages.txt',
          'val': src_id_root + 'devImages.txt',
          'test': src_id_root + 'testImages.txt'}

hp = {
    'batch_size': 16,
    'num_epochs': 5,
    'lr': 5e-5,
    'r': 16,
    'alpha': 32,
    'weight_decay': 0.01,
    'target_layers_replace': ['key', 'value', 'c_attn']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

collate_fn = u.return_collate_fn(tokenizer)
collate_fn_eval = u.collate_fn_eval

data_split = load_data.LoadDataSplit(
    src_id=src_id,
    src_cap=dir_Flickr_text,
    src_img=dir_Flickr_jpg,
    train_transforms=feature_extractor,
    val_transforms=feature_extractor,
    caption_mode='augmentation'
)

train_data, val_data, test_data = data_split.build()
train_dataloader = DataLoader(train_data, batch_size=hp['batch_size'],
                              shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=hp['batch_size'],
                            shuffle=False, collate_fn=collate_fn_eval)
test_dataloader = DataLoader(test_data, batch_size=hp['batch_size'],
                             shuffle=False, collate_fn=collate_fn_eval)

model.to(device)

for param in model.parameters():
    param.requires_grad = False

model = apply_lora(model, hp['target_layers_replace'], r=hp['r'], alpha=hp['alpha'])
model = model.to(device)
u.print_num_params(model)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=hp['lr'], weight_decay=hp['weight_decay'])

time_now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
path_save = f'save_dir/{time_now}/'
os.mkdir(path_save)
u.save_hyperparameters(path_save, hyperparams=hp)

train_cls = Trainer(num_epochs=hp['num_epochs'],
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    train_loader=train_dataloader,
                    val_loader=val_dataloader,
                    test_loader = test_dataloader,
                    path_save=path_save,
                    plot=False,
                    device=device)

train_cls()


