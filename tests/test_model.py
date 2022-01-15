from src.data.load_dataset import load_data
from tests.get_cfg import get_cfg
from src.models.model import MyAwesomeModel
def test_model():
    cfg = get_cfg()
    m = MyAwesomeModel(cfg)

    assert m.layers[0].in_channels == cfg.image.channels
    
