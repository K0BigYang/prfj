#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#%%
from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
import glob
from torch.utils.data import Dataset
import torch
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Test_DIR_Dataset(Dataset):
    def __init__(self, glob_path, transform):
        self.img_path_list = glob.glob(glob_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

@torch.inference_mode()
def main(checkpoint, glob_path, device='cuda'):
    model = load_from_checkpoint(checkpoint).eval().to(device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    for fname in args.blob_path:
        # Load image and prepare for input
        image = Image.open(fname).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{fname}: {pred[0]}')

#%% 
device='cuda'
checkpoint = 'last_100_epoch.ckpt'
model = load_from_checkpoint(checkpoint).eval().to(device)

# %%
from torch.utils.data import DataLoader
glob_path = 'data/test/*/*.png'
img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
test_dataset = Test_DIR_Dataset(glob_path, img_transform)
test_dataloader = DataLoader(test_dataset, batch_size=384, shuffle=False, num_workers=48)
# %%
results = []
for imgs in test_dataloader:
    # Load image and prepare for input
    imgs = imgs.to(device)
    p = model(imgs).softmax(-1)
    pred, p = model.tokenizer.decode(p)
    results+=pred.copy()
    #print(f'{fname}: {pred[0]}')

# %%
import pandas
sample_csv = './data/sample_submission.csv'
df_out = pandas.read_csv(sample_csv)
# %%
for path, pred in zip(test_dataset.img_path_list, results):
    filename = path.split('data/test/')[-1]
    df_out.loc[df_out['filename'] == filename, 'label'] = pred
# %%
df_out.to_csv('./data/411581003_100_epoch.csv', index=False)
# %%
