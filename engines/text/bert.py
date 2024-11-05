from engines.engine_base import Engine

from data.text.bert import TextDataConfig as DataConfig
from data.text.bert import TextDataManager as DataManager

from models.text.bert import BertConfig as ModelConfig
from models.text.bert import Bert as Model

from training.text import TrainConfig, Pipeline, forward


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