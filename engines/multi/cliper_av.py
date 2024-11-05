from engines.engine_base import Engine

from data.multi.cliper_av import MultiDataConfig as DataConfig
from data.multi.cliper_av import MultiDataManager as DataManager

from models.multi.cliper_av import CLIPerConfig as ModelConfig
from models.multi.cliper_av import CLIPer as Model

from training.multi import TrainConfig, Pipeline, forward_av


ENGINE = Engine(
    data_config_cls=DataConfig,
    data_manager_cls=DataManager,
    model_config_cls=ModelConfig,
    model_cls=Model,
    train_config_cls=TrainConfig,
    pipeline=Pipeline(forward_av)
)

def train(config, device, temp_dir=None):
    return ENGINE.train(config, device, temp_dir)

def test(path, device, split='test', epoch=None, load_best=False):
    return ENGINE.test(path, device, split, epoch, load_best)