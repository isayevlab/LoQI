from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from megalodon.models.module import Graph3DInterpolantModel


@pytest.mark.parametrize("config_path", [
    "scripts/conf/loqi/loqi_flow.yaml", "src/loqi/configs/loqi_flow.yaml",
])
@pytest.mark.parametrize("epoch", [0, 9, 10, 11])
@pytest.mark.parametrize("stage", ["train", "val"])
def test_flow_coordinate_loss_clamped_from_epoch_ten(config_path, epoch, stage):
    config = OmegaConf.load(Path(__file__).resolve().parents[1] / config_path)
    model = SimpleNamespace(
        loss_params=config.loss, current_epoch=epoch, global_variable="x",
        interpolants={"x": SimpleNamespace(loss_weight_t=torch.ones_like)},
        loss_fn=None, log=lambda *args, **kwargs: None,
    )
    model.loss_functions = Graph3DInterpolantModel.initialize_loss_functions(model)
    assert model.use_loss_clamps and model.loss_clamp_epoch_threshold == 10
    assert dict(model.loss_clamps) == {"x": 2.0}
    # Two molecules: per-molecule MSEs are 0.25 and 9.0. From epoch 10,
    # only the second is capped before averaging, with zero gradient.
    prediction = torch.tensor([[0.5] * 3] * 2 + [[3.0] * 3] * 2, requires_grad=True)
    batch = Data(batch=torch.tensor([0, 0, 1, 1]), x_target=torch.zeros(4, 3))
    loss, _ = Graph3DInterpolantModel.calculate_loss(
        model, batch, {"x_hat": prediction}, torch.tensor([0.5, 0.5]), stage,
    )
    clamped = epoch >= 10
    assert loss.item() == pytest.approx((0.25 + (2.0 if clamped else 9.0)) / 2)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:2]) == 6
    assert torch.count_nonzero(prediction.grad[2:]) == (0 if clamped else 6)
