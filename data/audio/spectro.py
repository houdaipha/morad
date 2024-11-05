import sys
import os
import logging
import yaml
from dataclasses import dataclass, field

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms._presets import ImageClassification

from data.utils import log_mel_spectrogram, pad_or_trim, spectrogram_to_image

logger = logging.getLogger(__name__)

@dataclass
class SpectrogramConfig:
    n_mels: int = 128
    sample_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 400
    img_size: int = 224
    normalize: bool = True
    transforms: bool = False
    crop_size: int | None = 224
    # target_length: int = None # Left for old code, should be removed

    def __post_init__(self):
        if not self.transforms:
            logger.info('No transforms applied, setting crop_size to None')
            self.crop_size = None
        self.target_length = (self.sample_rate // self.hop_length) * 5

@dataclass
class AudioDataConfig:
    path: str
    annotation_file: str # = 'morad.csv'
    spectorgram_config: SpectrogramConfig = field(
        default_factory=SpectrogramConfig)

    def __post_init__(self):
        if isinstance(self.spectorgram_config, dict):
            self.spectorgram_config = SpectrogramConfig(
                **self.spectorgram_config)

class AudioDataset(Dataset):
    def __init__(self, path, data, config: SpectrogramConfig):
        super().__init__()
        self.path = path
        self.data = data
        self.config = config

        if 'counts' in self.data.columns:
            self.weights = torch.tensor(1 / self.data['counts'].to_numpy())
        
        if self.config.transforms:
            self.transforms = ImageClassification(
                crop_size=self.config.crop_size,
                resize_size=self.config.img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        name = row['name']
        anno = row['emotion']

        path = os.path.join(self.path, 'audios', f'{name}.wav')
        spectorgram = self._read_audio(path)

        if self.config.transforms:
            spectorgram = self.transforms(spectorgram)

        sample = {
            'audio': spectorgram,
            'annotations': anno,
            'name': name}
        return sample

    def _read_audio(self, path):
        wave = log_mel_spectrogram(
            path, 
            n_mels=self.config.n_mels,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            n_fft=self.config.n_fft)
        wave = pad_or_trim(wave, self.config.target_length)
        wave = spectrogram_to_image(
            wave, self.config.img_size, normalize=self.config.normalize)

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
            config=self.config.spectorgram_config)
        logger.info(
            f'Dataset from split {split} loaded of length {len(dataset)}')
        return dataset

# Write a main function to test the AudioDataManager
def main():
    logging.basicConfig(level=logging.INFO)

    with open('configs/audio/mscnn.yaml', 'r') as f:
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