from engines.engine_base import Engine

from data.vision.base import VisionDataConfig as DataConfig
from data.vision.base import VisionDataManager as DataManager

from models.vision.videomae import VideoMAEConfig as ModelConfig
from models.vision.videomae import VideoMAE as Model

from training.vision import TrainConfig, Pipeline, forward


ENGINE = Engine(
    data_config_cls=DataConfig,
    data_manager_cls=DataManager,
    model_config_cls=ModelConfig,
    model_cls=Model,
    train_config_cls=TrainConfig,
    pipeline=Pipeline(forward)
)

def train(config, device, temp_dir=None):
    return ENGINE.train(config, device, temp_dir)

def test(path, device, split='test', epoch=None, load_best=False):
    return ENGINE.test(path, device, split, epoch, load_best)