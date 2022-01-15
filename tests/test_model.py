from src.data.load_dataset import load_data
from tests.get_cfg import get_cfg
from src.models.model import MyAwesomeModel
import torch

import pytest
def test_model():
    cfg = get_cfg()
    m = MyAwesomeModel(cfg)

    assert m.layers[0].in_channels == cfg.image.channels
    
def test_error_on_wrong_shape():
    cfg = get_cfg()
    m = MyAwesomeModel(cfg)
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        m(torch.randn(1,2,3))