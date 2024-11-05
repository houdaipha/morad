import sys
import os
import ast
import logging
import yaml
from PIL import Image
from dataclasses import dataclass, field
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, AutoTokenizer
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
    n_samples_per_token: int = hop_length * 2  # the initial convolutions has stride 2
    frames_per_second: int = exact_div(sample_rate, hop_length)  # 10ms per audio frame
    tokens_per_second: int = exact_div(sample_rate, n_samples_per_token)  # 20ms per audio token
    n_mels: int = 128  # Hardcoded for now

@dataclass
class TextBertConfig:
    tokenizer: str = 'UBC-NLP/MARBERTv2'
    max_length: int = 32
    add_special_tokens: bool = True

@dataclass
class MultiDataConfig:
    path: str
    annotation_file: str
    audio: AudioWhisperConfig = field(default_factory=AudioWhisperConfig)
    text: TextBertConfig = field(default_factory=TextBertConfig)

    def __post_init__(self):
        if isinstance(self.audio, dict):
            self.audio = AudioWhisperConfig(**self.audio)
        if isinstance(self.text, dict):
            self.text = TextBertConfig(**self.text)

# class ToTensor:
#     """Convert ndarrays in sample to Tensors."""

#     def __init__(self, img_size):
#         self.img_size = (img_size, img_size)

#     def __call__(self, sample):
#         trsfrm = transforms.Compose([
#             transforms.Resize(size=self.img_size),
#             transforms.PILToTensor(),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize(
#                 mean=(0.48145466, 0.4578275, 0.40821073), 
#                 std=(0.26862954, 0.26130258, 0.27577711))
#         ])
#         sframes = torch.stack([trsfrm(frame) for frame in sample])
#         return sframes

class MultiDataset(Dataset):
    def __init__(self, path, data, config):
        self.path = path
        self.data = data
        self.audio_config = config.audio
        self.text_config = config.text

        if 'counts' in self.data.columns:
            self.weights = torch.tensor(1 / self.data['counts'].to_numpy())

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.text_config.tokenizer)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        name = row['name']
        anno = row['emotion']
        text = row['Transcription']

        # Get audio
        audio_path = os.path.join(self.path, 'audios', f'{name}.wav')
        audio = self._read_audio(audio_path)

        # Get text
        tokens, mask = self._read_text(text)

        sample = {
            'audio': audio,
            'tokens': tokens,
            'attention_mask': mask,
            'annotations': anno,
            'name': name}

        return sample

    def _read_text(self, text):
        output = self.tokenizer(
            text,
            max_length=self.text_config.max_length,
            add_special_tokens=self.text_config.add_special_tokens,
            truncation=True,
            padding='max_length')
        tokens = torch.tensor(output['input_ids'])
        mask = torch.tensor(output['attention_mask'])
        return tokens, mask

    def _read_audio(self, path):
        audio = log_mel_spectrogram(
            path, 
            n_mels=self.audio_config.n_mels,
            sr=self.audio_config.sample_rate,
            hop_length=self.audio_config.hop_length,
            n_fft=self.audio_config.n_fft)
        audio = pad_or_trim(audio, self.audio_config.n_frames)
        return audio

class MultiDataManager:
    def __init__(self, config):
        self.config = config
        self.path = config.path

        df_path = os.path.join(self.path, 'annotations', config.annotation_file)
        self.df = pd.read_csv(df_path)

    def get_dataset(self, split, debug=False):
        data = self.df[self.df['split']==split][
            ['name', 'emotion', 'Transcription']]
        if split == 'train':
            # In case we want to oversample the minority classes
            counts = data['emotion'].value_counts()
            data['counts'] = data['emotion'].map(lambda x: counts[x])
        if debug:
            data = data[:10]

        dataset = MultiDataset(
            path=self.path,
            data=data,
            config=self.config)
        logger.info(
            f'Dataset from split {split} loaded of length {len(dataset)}')
        return dataset

def main():
    logging.basicConfig(level=logging.INFO)

    with open('configs/multi/cliper_at.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = MultiDataConfig(**config['data'])
    data = MultiDataManager(data_config)

    train_data = data.get_dataset('train', debug=True)
    val_data = data.get_dataset('val', debug=True)

    for d in train_data:
        print(d['audio'].shape)
        print(d['tokens'].shape)
        print(d['annotations'])
        break

    for d in val_data:
        print(d['audio'].shape)
        print(d['tokens'].shape)
        print(d['annotations'])
        break

if __name__ == '__main__':
    main()