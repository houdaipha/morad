import sys
import os
import logging
import yaml
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

logger = logging.getLogger(__name__)

@dataclass
class AudioDataConfig:
    path: str
    annotation_file: str # = 'morad.csv'
    target_length: int = 80640
    resample_rate: int = 16000

class AudioDataset(Dataset):
    def __init__(self, path, data, target_length, resample_rate):
        super().__init__()
        self.path = path
        self.data = data
        self.target_length = target_length
        self.resample_rate = resample_rate

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
        wave, sample_rate = torchaudio.load(path)
        wave = torchaudio.functional.resample(
            wave, sample_rate, self.resample_rate)

        if len(wave.shape) > 1:
            if wave.shape[0] == 1:
                wave = wave.squeeze()
            else:
                wave = wave.mean(axis=0)  # multiple channels, average

        target_length = self.target_length
        if wave.size(0) > target_length:
            wave = wave[:, :target_length]
        else:
            padding = torch.zeros(target_length - wave.size(0))
            wave = torch.cat((wave, padding), dim=0)

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
            target_length=self.config.target_length,
            resample_rate=self.config.resample_rate)
        logger.info(
            f'Dataset from split {split} loaded of length {len(dataset)}')
        return dataset

# Write a main function to test the AudioDataManager
def main():
    logging.basicConfig(level=logging.INFO)

    config = AudioDatasetConfig(
        path='/morad_dir/Datasets/EMAD/',
        fold_id=1,
        annotation='morad.csv')
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