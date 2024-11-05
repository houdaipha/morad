import sys
import os
import ast
import logging
import yaml
import av

from typing import Optional

from PIL import Image
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import VivitImageProcessor, AutoImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class VisionDataConfig:
    path: str
    annotation_file: str
    frames: int
    processor: str = 'google/vivit-b-16x2-kinetics400'
    image_size: int = 224

class VisionDataset(Dataset):
    def __init__(self, path, data, num_frames, image_size, image_processor):
        self.path = path
        self.data = data
        self.num_frames = num_frames

        if 'counts' in self.data.columns:
            self.weights = torch.tensor(1 / self.data['counts'].to_numpy())

        if image_processor == 'google/vivit-b-16x2-kinetics400':
            self.image_processor = VivitImageProcessor.from_pretrained(
                pretrained_model_name_or_path=image_processor,
                size={'height': image_size, 'width': image_size},
            )
            logger.info('Loaded Vivit image processor')
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path=image_processor,
                size={'height': image_size, 'width': image_size},
            )
            logger.info('Loaded Auto image processor')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        name = row['name']
        anno = row['emotion']
        faces = row['faces']

        path = os.path.join(self.path, 'frames', name)

        frames = self._read_frames(path, faces)
        frames = self.image_processor(list(frames), return_tensors='pt')
        frames = frames['pixel_values'].squeeze(0)

        sample = {
            'frames': frames,
            'annotations': anno,
            'name': name}
        return sample

    def _read_frames(self, path, faces):
        # NOTE: Sorting for consistency
        faces = ast.literal_eval(faces)
        faces.sort(
            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        step_size = len(faces) // self.num_frames
        faces = faces[::step_size][:self.num_frames]

        frames = []
        for face in faces:
            face_path = os.path.join(path, face)
            container = av.open(face_path)
            gen = container.decode(video=0)
            frame = next(gen).to_ndarray(format='rgb24')
            frames.append(frame)
        frames = np.stack(frames)
        return frames

class VisionDataManager:
    def __init__(self, config: VisionDataConfig):
        self.config = config
        self.path = config.path

        df_path = os.path.join(
            config.path, 'annotations', config.annotation_file)
        self.df = pd.read_csv(df_path)

    def get_dataset(self, split, debug=False):
        data = self.df[self.df['split']==split][['name', 'emotion', 'faces']]

        if split == 'train':
            counts = data['emotion'].value_counts()
            data['counts'] = data['emotion'].map(lambda x: counts[x])

        if debug:
            data = data[:10]

        dataset = VisionDataset(
            path=self.path, 
            data=data, 
            num_frames=self.config.frames, 
            image_size=self.config.image_size, 
            image_processor=self.config.processor)
        logger.info(
            f'Loaded {split} dataset with {len(dataset)} samples')
        return dataset

# Write a main function to test the VisionDataManager
def main():
    logging.basicConfig(level=logging.INFO)
    config_path = '/home/houdaifa.atou/main/code/morad/configs/vision/vivit.yaml'

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = VisionDataConfig(**config['data'])

    manager = VisionDataManager(config)
    train_dataset = manager.get_dataset('train', debug=True)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for i, batch in enumerate(train_loader):
        logger.info(f'Batch {i} loaded')
        logger.info(batch['frames'].shape)
        if i == 2:
            break

if __name__ == '__main__':
    main()