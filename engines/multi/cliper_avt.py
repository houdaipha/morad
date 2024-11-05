from engines.engine_base import Engine

from data.multi.cliper_avt import MultiDataConfig as DataConfig
from data.multi.cliper_avt import MultiDataManager as DataManager

from models.multi.cliper_avt import CLIPerConfig as ModelConfig
from models.multi.cliper_avt import CLIPer as Model

from training.multi import TrainConfig, Pipeline, forward_avt


ENGINE = Engine(
    data_config_cls=DataConfig,
    data_manager_cls=DataManager,
    model_config_cls=ModelConfig,
    model_cls=Model,
    train_config_cls=TrainConfig,
    pipeline=Pipeline(forward_avt)
)

def train(config, device, temp_dir=None):
    return ENGINE.train(config, device, temp_dir)

def test(path, device, split='test', epoch=None, load_best=False):
    return ENGINE.test(path, device, split, epoch, load_best)