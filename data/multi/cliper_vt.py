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

logger = logging.getLogger(__name__)


@dataclass
class VisionClipConfig:
    frames: int = 16
    image_size: int = 224

@dataclass
class TextBertConfig:
    tokenizer: str = 'UBC-NLP/MARBERTv2'
    max_length: int = 32
    add_special_tokens: bool = True

@dataclass
class MultiDataConfig:
    path: str
    annotation_file: str
    vision: VisionClipConfig = field(default_factory=VisionClipConfig)
    text: TextBertConfig = field(default_factory=TextBertConfig)

    def __post_init__(self):
        if isinstance(self.vision, dict):
            self.vision = VisionClipConfig(**self.vision)
        if isinstance(self.text, dict):
            self.text = TextBertConfig(**self.text)

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size):
        self.img_size = (img_size, img_size)

    def __call__(self, sample):
        trsfrm = transforms.Compose([
            transforms.Resize(size=self.img_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711))
        ])
        sframes = torch.stack([trsfrm(frame) for frame in sample])
        return sframes

class MultiDataset(Dataset):
    def __init__(self, path, data, config, transform=None):
        self.path = path
        self.data = data
        self.vision_config = config.vision
        self.text_config = config.text
        self.transform = transform

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
        faces = row['faces']
        text = row['Transcription']

        # Get frames
        frames_path = os.path.join(self.path, 'frames', name)
        frames = self._read_frames(frames_path, faces)
        if self.transform:
            frames = self.transform(frames)

        # Get text
        tokens, mask = self._read_text(text)

        sample = {
            'frames': frames,
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

    def _read_frames(self, path, faces):
        # XXX: Sorting for consistency
        faces = ast.literal_eval(faces)
        faces.sort(
            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        step_size = len(faces) // self.vision_config.frames
        faces = faces[::step_size][:self.vision_config.frames]

        frames = []
        for face in faces:
            face_path = os.path.join(path, face)
            try:
                p = Image.open(face_path).convert('RGB')
            except Exception as e:
                logger.error(
                    f'Exception {e} while reading frame at: {path}')
            else:
                frames.append(p)
        return frames

class MultiDataManager:
    def __init__(self, config):
        self.config = config
        self.path = config.path

        df_path = os.path.join(self.path, 'annotations', config.annotation_file)
        self.df = pd.read_csv(df_path)

    def get_dataset(self, split, debug=False):
        data = self.df[self.df['split']==split][
            ['name', 'emotion', 'faces', 'Transcription']]
        if split == 'train':
            # In case we want to oversample the minority classes
            counts = data['emotion'].value_counts()
            data['counts'] = data['emotion'].map(lambda x: counts[x])
        if debug:
            data = data[:10]

        transform = ToTensor(self.config.vision.image_size)
        dataset = MultiDataset(
            path=self.path,
            data=data,
            config=self.config,
            transform=transform)
        logger.info(
            f'Dataset from split {split} loaded of length {len(dataset)}')
        return dataset

def main():
    logging.basicConfig(level=logging.INFO)

    with open('configs/multi/clipere_t.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = MultiDataConfig(**config['data'])
    data = MultiDataManager(data_config)

    train_data = data.get_dataset('train', debug=True)
    val_data = data.get_dataset('val', debug=True)

    for d in train_data:
        print(d['frames'].shape)
        print(d['tokens'].shape)
        print(d['annotations'])
        break

    for d in val_data:
        print(d['frames'].shape)
        print(d['tokens'].shape)
        print(d['annotations'])
        break

if __name__ == '__main__':
    main()