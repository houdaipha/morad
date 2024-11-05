import sys
import os
import ast
import logging
import yaml
from PIL import Image
from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

@dataclass
class VisionDataConfig:
    path: str
    annotation_file: str
    frames: int
    image_size: int = 224

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

class VisionDataset(Dataset):
    def __init__(self, path, data, num_frames, transform=None):
        self.path = path
        self.data = data
        self.num_frames = num_frames
        self.transform = transform

        if 'counts' in self.data.columns:
            self.weights = torch.tensor(1 / self.data['counts'].to_numpy())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        name = row['name']
        anno = row['emotion']
        faces = row['faces']

        path = os.path.join(self.path, 'frames', name)
        # frames = read_frames(path, faces, self.num_frames)
        frames = self._read_frames(path, faces)


        if self.transform:
            frames = self.transform(frames)
        sample = {
            'frames': frames,
            'annotations': anno,
            'name': name}
        return sample

    def _read_frames(self, path, faces):
        # XXX: Sorting for consistency
        faces = ast.literal_eval(faces)
        faces.sort(
            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        step_size = len(faces) // self.num_frames
        faces = faces[::step_size][:self.num_frames]

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

        transform = ToTensor(self.config.image_size)

        dataset = VisionDataset(
            self.path, data, self.config.frames, transform)
        logger.info(
            f'Loaded {split} dataset with {len(dataset)} samples')
        return dataset

# Write a main function to test the VisionDataManager
def main():
    logging.basicConfig(level=logging.INFO)
    config_path = '/home/houdaifa.atou/main/code/morad/configs/clip.yaml'

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