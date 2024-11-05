import sys
import os
import logging
import yaml
from dataclasses import dataclass, field

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from data.utils import load_mfcc

logger = logging.getLogger(__name__)

@dataclass
class MFCCConfig:
    n_mfcc: int = 13
    n_mels: int = 128 
    sample_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 400
    deltas: bool = False
    target_length: int = None # Left for old code, should be removed

    def __post_init__(self):
        self.target_length = (self.sample_rate // self.hop_length) * 5

@dataclass
class AudioDataConfig:
    path: str
    annotation_file: str # = 'morad.csv'
    mfcc_config: MFCCConfig = field(
        default_factory=MFCCConfig)

    def __post_init__(self):
        if isinstance(self.mfcc_config, dict):
            self.mfcc_config = MFCCConfig(**self.mfcc_config)

class AudioDataset(Dataset):
    def __init__(self, path, data, config: MFCCConfig):
        super().__init__()
        self.path = path
        self.data = data
        self.config = config

        if 'counts' in self.data.columns:
            self.weights = torch.tensor(1 / self.data['counts'].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        name = row['name']
        anno = row['emotion']

        path = os.path.join(self.path, 'audios', f'{name}.wav')
        audio = self._read_audio(path)

        sample = {
            'audio': audio,
            'annotations': anno,
            'name': name}
        return sample

    def _read_audio(self, path):
        wave = load_mfcc(
            path,
            target_length=self.config.target_length,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            deltas=self.config.deltas
        )

        return wave

class AudioDataManager:
    def __init__(self, config) -> None:
        self.config = config

        self.path = config.path

        df_path = os.path.join(config.path, 'annotations', config.annotation_file)
        self.df = pd.read_csv(df_path)

    def get_dataset(self, split, debug=False):
        data = self.df[self.df['split']==split][['name', 'emotion']]
        if split == 'train':
            # In case we want to oversample the minority classes
            counts = data['emotion'].value_counts()
            data['counts'] = data['emotion'].map(lambda x: counts[x])
        if debug:
            data = data[:10]
        dataset = AudioDataset(
            path=self.path,
            data=data,
            config=self.config.mfcc_config)
        logger.info(
            f'Dataset from split {split} loaded of length {len(dataset)}')
        return dataset

# Write a main function to test the AudioDataManager
def main():
    logging.basicConfig(level=logging.INFO)

    with open('configs/audio/mflstm.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = AudioDataConfig(**config['data'])
    data = AudioDataManager(config)

    train_dataset = data.get_dataset('train', debug=True)
    val_dataset = data.get_dataset('val', debug=True)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    for batch in train_loader:
        logger.info(batch['audio'].shape)
        logger.info(batch['annotations'])
        break
    for batch in val_loader:
        logger.info(batch['audio'].shape)
        logger.info(batch['annotations'])
        break

if __name__ == '__main__':
    main()