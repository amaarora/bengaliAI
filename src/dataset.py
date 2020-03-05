import pandas as pd
import torch
import os 
import glob
from tqdm import tqdm 
from PIL import Image
import numpy as np
import joblib
import albumentations
from pathlib import Path

class BengaliDataset():
    def __init__(self, folds, data_path="../data", height=137, width=236):
        data_path = Path(data_path)
        df = pd.read_csv(data_path/'train_folds.csv')
        df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'kfold']]
        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.df = df
        self.data_path = data_path
        self.img_ids = df['image_id'].values
        self.grapheme_root = df['grapheme_root'].values
        self.vowel_diacritic = df['vowel_diacritic'].values
        self.consonant_diacritic = df['consonant_diacritic'].values
        self.h = height
        self.w = width
        
        if len(folds) == 1: 
            self.aug = albumentations.Compose([
                albumentations.Resize(self.h, self.w, always_apply=True), 
                albumentations.Normalize(always_apply=True) #uses imagenette stats by default
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(self.h, self.w, always_apply=True),
                albumentations.ShiftScaleRotate(rotate_limit=5, p=0.7),
                albumentations.Normalize(always_apply=True) #uses imagenette stats by default
            ])
            
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        image = joblib.load(f"{self.data_path}/image_pickles/{self.img_ids[idx]}.pkl")
        image = image.reshape(self.h, self.w).astype(float)
        image = Image.fromarray(image).convert("RGB")
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'grapheme_root': torch.tensor(self.grapheme_root[idx], dtype=torch.long), 
            'vowel_diacritic': torch.tensor(self.vowel_diacritic[idx], dtype=torch.long), 
            'consonant_diacritic': torch.tensor(self.consonant_diacritic[idx], dtype=torch.long), 
        }