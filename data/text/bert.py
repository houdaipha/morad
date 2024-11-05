import os
import logging
import yaml

from typing import Optional
from dataclasses import dataclass

import pandas as pd
import torch

from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

@dataclass
class TextDataConfig:
    path: str
    annotation_file: str
    tokenizer: str = 'UBC-NLP/MARBERT'
    max_length: int = 32
    add_special_tokens: bool = True

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, add_special_tokens=False):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        name = row['name']
        anno = row['emotion']
        text = row['Transcription']

        output = self.tokenizer(
            text,
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            padding='max_length')
        tokens = output['input_ids']
        attention_mask = output['attention_mask']

        sample = {
            'name': name,
            'tokens': torch.tensor(tokens),
            'attention_mask': torch.tensor(attention_mask),
            'annotations': anno
        }
        return sample

class TextDataManager:
    def __init__(self, config: TextDataConfig):
        self.config = config
        self.path = config.path

        df_path = os.path.join(
            config.path, 'annotations', config.annotation_file)
        self.df = pd.read_csv(df_path)

    def get_dataset(self, split, debug=False):
        data = self.df[self.df['split']==split][
            ['name', 'emotion', 'Transcription']]

        if split == 'train':
            counts = data['emotion'].value_counts()
            data['counts'] = data['emotion'].map(lambda x: counts[x])

        if debug:
            data = data[:10]

        dataset = TextDataset(
            data=data, 
            tokenizer=self.config.tokenizer, 
            max_length=self.config.max_length,
            add_special_tokens=self.config.add_special_tokens)
        
        logger.info(f'Loaded {split} dataset with {len(dataset)} samples')

        return dataset

def main():
    logging.basicConfig(level=logging.INFO)

    config_path = '/home/houdaifa.atou/main/code/morad/configs/text/roberta.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config = TextDataConfig(**config['data'])

    manager = TextDataManager(config)
    train_dataset = manager.get_dataset('train', debug=True)
    val_dataset = manager.get_dataset('val', debug=True)
    test_dataset = manager.get_dataset('test', debug=True)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    for batch in train_loader:
        logger.info(batch['tokens'].shape)
        logger.info(batch['tokens'])
        break
    for batch in val_loader:
        logger.info(batch['tokens'].shape)
        break
    for batch in test_loader:
        logger.info(batch['tokens'].shape)
        break

if __name__ == '__main__':
    main()
