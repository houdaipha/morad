import os
import logging
import yaml
from dataclasses import dataclass, field
import pandas as pd
import torch
from torch.utils.data import Dataset
from data.utils import pad_or_trim, log_mel_spectrogram

logger = logging.getLogger(__name__)

def exact_div(x, y):
    assert x % y == 0
    return x // y

@dataclass
class AudioWhisperConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    chunk_length: int = 30
    n_samples: int = chunk_length * sample_rate  # 480000 samples in a 30-second chunk
    n_frames: int = exact_div(n_samples, hop_length)  # 3000 frames in a mel spectrogram input
    n_mels: int = 128  # Hardcoded for now
    # Unrelated to the audio processing
    n_samples_per_token: int = hop_length * 2  # the initial convolutions has stride 2
    frames_per_second: int = exact_div(sample_rate, hop_length)  # 10ms per audio frame
    tokens_per_second: int = exact_div(sample_rate, n_samples_per_token)  # 20ms per audio token


@dataclass
class AudioDataConfig:
    path: str
    annotation_file: str
    whisper_config: AudioWhisperConfig = field(default_factory=AudioWhisperConfig)

    def __post_init__(self):
        if isinstance(self.whisper_config, dict):
            self.whisper_config = AudioWhisperConfig(**self.whisper_config)

class AudioDataset(Dataset):
    def __init__(self, path, data, config: AudioWhisperConfig):
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

        audio = log_mel_spectrogram(
            path, 
            n_mels=self.config.n_mels,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            n_fft=self.config.n_fft)
        audio = pad_or_trim(audio, self.config.n_frames)

        sample = {
            'audio': audio,
            'annotations': anno,
            'name': name}
        return sample

class AudioDataManager:
    def __init__(self, config) -> None:
        self.config = config
        
        self.path = config.path

        df_path = os.path.join(self.path, 'annotations', config.annotation_file)
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
            config=self.config.whisper_config)
        logger.info(
            f'Dataset from split {split} loaded of length {len(dataset)}')
        return dataset

def main():
    logging.basicConfig(level=logging.INFO)

    with open('configs/audio/whisper.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = AudioDataConfig(**config['data'])
    data = AudioDataManager(data_config)

    train_data = data.get_dataset('train', debug=True)
    val_data = data.get_dataset('val', debug=True)

    for d in train_data:
        print(d['audio'].shape)
        print(d['annotations'])
        break

    for d in val_data:
        print(d['audio'].shape)
        print(d['annotations'])
        break

if __name__ == '__main__':
    main()
