from types import SimpleNamespace

import torch
from torch import nn

from scripts.run_raw_cifar_tta import RawState
from scripts.train_cifar import build_model
from shiftroute.adapters import PriorAdapterConfig
from shiftroute.router import Action


def first_batch_norm(model):
    return next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d))


def make_state(name: str) -> RawState:
    model = build_model(10).eval()
    source_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return RawState(
        name=name,
        source_model=model,
        source_state=source_state,
        source_prior=torch.full((10,), 0.1),
        classes=10,
        tent_learning_rate=1e-3,
        eta_entropy_margin=2.0,
        eta_diversity_margin=1.1,
        sar_reset_constant=0.0,
        ods_learning_rate=1e-4,
        ods_queue_size=64,
        ods_knn=5,
        seed=2026,
        prior_config=PriorAdapterConfig(),
    )


def test_freeze_uses_source_bn_and_representation_state_is_reversible():
    images = torch.randn(8, 3, 32, 32)
    diagnosis = SimpleNamespace(target_prior=torch.full((10,), 0.1))
    for name in ("prequential_tent", "prequential_eta", "prequential_sar", "prequential_deyo"):
        state = make_state(name)
        source_bn = first_batch_norm(state.model)
        assert source_bn.track_running_stats
        assert source_bn.running_mean is not None
        frozen_logits = state.predict(images)
        assert frozen_logits.shape == (8, 10)

        state.transition(Action.REPRESENTATION_ONLY, diagnosis, images)
        adapted_bn = first_batch_norm(state.model)
        assert state.optimizer is not None
        assert not adapted_bn.track_running_stats
        assert adapted_bn.running_mean is None

        state.transition(Action.FREEZE, diagnosis, images)
        restored_bn = first_batch_norm(state.model)
        assert state.optimizer is None
        assert restored_bn.track_running_stats
        assert restored_bn.running_mean is not None


def test_prior_only_never_enables_normalization_adaptation():
    images = torch.randn(4, 3, 32, 32)
    target_prior = torch.linspace(1, 10, 10, dtype=torch.float32)
    target_prior /= target_prior.sum()
    diagnosis = SimpleNamespace(target_prior=target_prior)
    state = make_state("always_prior")
    state.transition(Action.PRIOR_ONLY, diagnosis, images)
    assert state.optimizer is None
    assert first_batch_norm(state.model).track_running_stats
    assert state.action == Action.PRIOR_ONLY


def test_deyo_patch_shuffle_preserves_shape_and_per_image_values():
    images = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32)
    shuffled = RawState.patch_shuffle(images)
    assert shuffled.shape == images.shape
    for index in range(len(images)):
        assert torch.equal(images[index].flatten().sort().values, shuffled[index].flatten().sort().values)
