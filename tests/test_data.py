from src.data.load_dataset import load_data
from tests.get_cfg import get_cfg

def test_data():
    cfg = get_cfg()
    trainloader, testloader = load_data(cfg, 1)

    assert len(trainloader) == 30000
    assert len(testloader) == 5000


