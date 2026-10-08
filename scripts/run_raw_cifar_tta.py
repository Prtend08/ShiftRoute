"""Prequential raw-image CIFAR TTA with ShiftRoute-gated Tent updates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets

from shiftroute.adapters import PriorAdapterConfig, prior_corrected_logits
from shiftroute.diagnostics import diagnose_window, fit_source_statistics
from shiftroute.experiment import (
    apply_prior_reliability_guard,
    FeatureBank,
    StreamPhase,
    build_stream,
    calibrate_from_clean_windows,
    development_prior_reliability_from_clean_windows,
    development_scales_from_clean_windows,
    expected_calibration_error,
    long_tailed_prior,
)
from shiftroute.router import Action, HysteresisRouter
try:
    from train_cifar import build_model, dataset_class
except ModuleNotFoundError:  # importable as scripts.run_raw_cifar_tta in tests
    from scripts.train_cifar import build_model, dataset_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--corruption-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--representation-a", type=Path, required=True)
    parser.add_argument("--representation-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--phase-windows", type=int, default=6)
    parser.add_argument("--imbalance-ratio", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tent-learning-rate", type=float, default=1e-3)
    parser.add_argument("--eta-entropy-margin", type=float, default=0.4)
    parser.add_argument("--eta-diversity-margin", type=float)
    parser.add_argument("--sar-reset-constant", type=float, default=0.2)
    parser.add_argument("--ods-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ods-queue-size", type=int, default=64)
    parser.add_argument("--ods-knn", type=int, default=None)
    parser.add_argument("--method-set", choices=["full", "ods_router"], default="full")
    parser.add_argument("--prior-strength", type=float, default=0.5)
    parser.add_argument("--prior-momentum", type=float, default=0.5)
    parser.add_argument("--prior-ratio-clip", type=float, default=5.0)
    args = parser.parse_args()
    if args.ods_knn is None:
        args.ods_knn = 5 if args.dataset == "cifar10" else 2
    return args


def load_bank(path: Path, name: str) -> tuple[FeatureBank, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return (
        FeatureBank(
            name=name,
            features=artifact["features"].float(),
            logits=artifact["logits"].float(),
            labels=artifact["labels"].long(),
        ),
        artifact,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_normalization_adaptation(model: nn.Module):
    model.eval()
    model.requires_grad_(False)
    parameters = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            # Official Tent convention: use current-batch statistics without
            # accumulating test information in running buffers.
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            parameters.extend([module.weight, module.bias])
    if not parameters:
        raise RuntimeError("normalization adaptation requires affine BatchNorm2d parameters")
    return parameters


def configure_tent(model: nn.Module, learning_rate: float):
    parameters = configure_normalization_adaptation(model)
    return torch.optim.Adam(parameters, lr=learning_rate)


def configure_ods(model: nn.Module, learning_rate: float):
    """Official ODS convention: affine BN updates with learned statistics."""

    model.eval()
    model.requires_grad_(False)
    parameters = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = True
            module.momentum = 0.01
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            parameters.extend([module.weight, module.bias])
    if not parameters:
        raise RuntimeError("ODS requires affine BatchNorm2d parameters")
    return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)


def resnet_logits_features(
    model: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return torchvision-ResNet logits and penultimate pooled features."""

    values = model.conv1(images)
    values = model.bn1(values)
    values = model.relu(values)
    values = model.maxpool(values)
    values = model.layer1(values)
    values = model.layer2(values)
    values = model.layer3(values)
    values = model.layer4(values)
    features = torch.flatten(model.avgpool(values), 1)
    return model.fc(features), features


@torch.no_grad()
def ods_lame_refine(
    logits: torch.Tensor,
    features: torch.Tensor,
    *,
    knn: int,
    maximum_steps: int = 100,
) -> torch.Tensor:
    """LAME refinement used by the released ODS implementation."""

    normalized = F.normalize(features.float(), p=2, dim=1)
    distances = torch.cdist(normalized, normalized, p=2)
    neighbors = min(int(knn), len(normalized))
    sigma = distances.topk(k=neighbors, dim=1, largest=False).values[:, -1].mean()
    sigma = sigma.clamp_min(1e-12)
    kernel = torch.exp(-(distances.square()) / (2.0 * sigma.square()))
    kernel = 0.5 * (kernel + kernel.t())
    unary = -torch.log_softmax(logits.float(), dim=1)
    probabilities = (-unary).softmax(dim=1)
    previous_energy = float("inf")
    for _ in range(maximum_steps):
        pairwise = kernel.matmul(probabilities)
        probabilities = (-unary + pairwise).softmax(dim=1)
        energy = float(
            (
                unary * probabilities
                - pairwise * probabilities
                + probabilities * probabilities.clamp_min(1e-20).log()
            ).sum()
        )
        if np.isfinite(previous_energy) and (
            abs(energy - previous_energy) <= 1e-8 * abs(previous_energy)
        ):
            break
        previous_energy = energy
    return probabilities


class SAM(torch.optim.Optimizer):
    """First-order sharpness-aware optimizer used by the official SAR method."""

    def __init__(self, params, learning_rate: float, momentum: float = 0.9, rho: float = 0.05):
        defaults = dict(lr=learning_rate, momentum=momentum, rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = torch.optim.SGD(self.param_groups, lr=learning_rate, momentum=momentum)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self) -> None:
        norms = [parameter.grad.norm(2) for group in self.param_groups for parameter in group["params"] if parameter.grad is not None]
        if not norms:
            return
        shared_device = self.param_groups[0]["params"][0].device
        grad_norm = torch.norm(torch.stack([value.to(shared_device) for value in norms]), p=2)
        for group in self.param_groups:
            scale = group.get("rho", 0.05) / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_parameter"] = parameter.data.clone()
                parameter.add_(parameter.grad * scale.to(parameter))
        self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, update: bool = True) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                old_parameter = self.state[parameter].pop("old_parameter", None)
                if old_parameter is not None:
                    parameter.data.copy_(old_parameter)
        if update:
            self.base_optimizer.step()
        self.zero_grad(set_to_none=True)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    return -(probabilities * logits.log_softmax(dim=1)).sum(dim=1)


def selected_action(name: str, truth: Action, routed: Action) -> Action:
    return {
        "source": Action.FREEZE,
        "always_prior": Action.PRIOR_ONLY,
        "prequential_tent": Action.REPRESENTATION_ONLY,
        "prequential_eta": Action.REPRESENTATION_ONLY,
        "prequential_sar": Action.REPRESENTATION_ONLY,
        "prequential_deyo": Action.REPRESENTATION_ONLY,
        "prequential_ods": Action.REPRESENTATION_ONLY,
        "shiftroute_ods": routed,
        "prequential_joint": Action.JOINT,
        "shiftroute_tent": routed,
        "oracle_tent": truth,
    }[name]


class RawState:
    def __init__(
        self,
        name: str,
        source_model: nn.Module,
        source_state: Dict[str, torch.Tensor],
        source_prior: torch.Tensor,
        classes: int,
        tent_learning_rate: float,
        eta_entropy_margin: float,
        eta_diversity_margin: float,
        sar_reset_constant: float,
        ods_learning_rate: float,
        ods_queue_size: int,
        ods_knn: int,
        seed: int,
        prior_config: PriorAdapterConfig,
    ) -> None:
        self.name = name
        self.source_model = copy.deepcopy(source_model).eval()
        self.model = copy.deepcopy(self.source_model).eval()
        self.source_state = source_state
        self.tent_learning_rate = tent_learning_rate
        self.optimizer = None
        self.adaptation_kind = (
            "eta"
            if name == "prequential_eta"
            else "sar"
            if name == "prequential_sar"
            else "deyo"
            if name == "prequential_deyo"
            else "ods"
            if name in {"prequential_ods", "shiftroute_ods"}
            else "tent"
        )
        self.entropy_margin = eta_entropy_margin * float(np.log(classes))
        self.diversity_margin = eta_diversity_margin
        self.sar_reset_constant = sar_reset_constant
        self.ods_learning_rate = ods_learning_rate
        self.ods_queue_size = ods_queue_size
        self.ods_knn = ods_knn
        self.ods_rng = np.random.default_rng(seed)
        self.ods_memory: torch.Tensor | None = None
        self.ods_memory_labels: list[int] = []
        self.ods_memory_positions: list[list[int]] = [[] for _ in range(classes)]
        self.ods_seen_per_class = np.zeros(classes, dtype=np.int64)
        self.ods_label_queue = np.full(ods_queue_size, -1, dtype=np.int64)
        self.ods_queue_pointer = 0
        self.ods_pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self.current_model_probs = None
        self.sar_entropy_ema = None
        self.ods_memory = None
        self.ods_memory_labels = []
        self.ods_memory_positions = [[] for _ in range(len(self.ods_seen_per_class))]
        self.ods_seen_per_class.fill(0)
        self.ods_label_queue.fill(-1)
        self.ods_queue_pointer = 0
        self.ods_pending = None
        self.action = Action.FREEZE
        self.target_prior = source_prior.clone()
        self.source_prior = source_prior
        self.prior_config = prior_config

    def reset_representation(self) -> None:
        self.model = copy.deepcopy(self.source_model).eval()
        self.optimizer = None
        self.current_model_probs = None
        self.sar_entropy_ema = None

    def enable_representation(self) -> None:
        self.model = copy.deepcopy(self.source_model).eval()
        if self.adaptation_kind == "ods":
            self.optimizer = configure_ods(self.model, self.ods_learning_rate)
        elif self.adaptation_kind == "sar":
            parameters = configure_normalization_adaptation(self.model)
            self.optimizer = SAM(parameters, learning_rate=self.tent_learning_rate)
        else:
            self.optimizer = configure_tent(self.model, self.tent_learning_rate)

    @torch.no_grad()
    def ods_outputs(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with the released ODS geometric-mean/LAME refinement."""

        self.model.eval()
        adapted_logits, adapted_features = resnet_logits_features(self.model, images)
        base_logits = self.source_model(images)
        optimized_distribution = ods_lame_refine(
            base_logits, adapted_features, knn=self.ods_knn
        )
        probabilities = torch.sqrt(
            adapted_logits.softmax(dim=1) * optimized_distribution
        )
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return probabilities.clamp_min(1e-10).log(), probabilities.argmax(dim=1)

    def ods_store_sample(self, image: torch.Tensor, label: int) -> None:
        """Pseudo-label-balanced reservoir from the official ODS release."""

        image = image.detach().cpu().unsqueeze(0)
        self.ods_seen_per_class[label] += 1
        if self.ods_memory is None:
            self.ods_memory = image
            self.ods_memory_labels = [label]
            self.ods_memory_positions[label].append(0)
            return
        if len(self.ods_memory_labels) < self.ods_queue_size:
            position = len(self.ods_memory_labels)
            self.ods_memory = torch.cat((self.ods_memory, image), dim=0)
            self.ods_memory_labels.append(label)
            self.ods_memory_positions[label].append(position)
            return

        sizes = np.asarray([len(values) for values in self.ods_memory_positions])
        largest = np.flatnonzero(sizes == sizes.max()).tolist()
        if label not in largest:
            replaced_class = int(self.ods_rng.choice(largest))
            replaced_position = int(self.ods_rng.choice(self.ods_memory_positions[replaced_class]))
            self.ods_memory_positions[replaced_class].remove(replaced_position)
        else:
            candidate = int(self.ods_rng.integers(0, self.ods_seen_per_class[label]))
            if candidate >= len(self.ods_memory_positions[label]):
                return
            replaced_position = self.ods_memory_positions[label][candidate]
            self.ods_memory_positions[label].remove(replaced_position)
        self.ods_memory[replaced_position] = image[0]
        self.ods_memory_labels[replaced_position] = label
        self.ods_memory_positions[label].append(replaced_position)

    def adapt_ods(self, images: torch.Tensor) -> None:
        if self.ods_pending is None:
            _, labels = self.ods_outputs(images)
            stored_images = images.detach().cpu()
        else:
            stored_images, labels = self.ods_pending
        labels_cpu = labels.detach().cpu().long()
        for image, label in zip(stored_images, labels_cpu.tolist()):
            self.ods_store_sample(image, int(label))
            self.ods_label_queue[self.ods_queue_pointer] = int(label)
            self.ods_queue_pointer = (self.ods_queue_pointer + 1) % self.ods_queue_size
        self.ods_pending = None
        if self.ods_memory is None:
            return

        class_counts = np.bincount(
            self.ods_label_queue[self.ods_label_queue >= 0],
            minlength=len(self.ods_seen_per_class),
        ).astype(np.float32)
        distribution = torch.from_numpy(class_counts / self.ods_queue_size)
        class_weights = 1.0 - distribution + 0.1
        class_weights = class_weights / class_weights.sum()
        memory_images = self.ods_memory.to(images.device, non_blocking=True)
        memory_labels = torch.tensor(
            self.ods_memory_labels, device=images.device, dtype=torch.long
        )
        sample_weights = class_weights.to(images.device)[memory_labels]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        memory_logits = self.model(memory_images)
        loss = (entropy(memory_logits) * sample_weights).sum() / sample_weights.sum()
        loss.backward()
        self.optimizer.step()

    def adapt_eta(self, images: torch.Tensor) -> None:
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        entropies = entropy(logits)
        reliable = entropies < self.entropy_margin
        selected_logits = logits[reliable]
        selected_entropies = entropies[reliable]
        if self.current_model_probs is not None and len(selected_logits):
            similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0), selected_logits.softmax(1), dim=1
            ).abs()
            diverse = similarities < self.diversity_margin
            selected_logits = selected_logits[diverse]
            selected_entropies = selected_entropies[diverse]
        if len(selected_logits):
            weights = torch.exp(self.entropy_margin - selected_entropies.detach())
            (selected_entropies * weights).mean().backward()
            self.optimizer.step()
            with torch.no_grad():
                batch_probs = selected_logits.softmax(1).mean(0)
                self.current_model_probs = (
                    batch_probs
                    if self.current_model_probs is None
                    else 0.9 * self.current_model_probs + 0.1 * batch_probs
                )

    def adapt_sar(self, images: torch.Tensor) -> None:
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        entropies = entropy(logits)
        reliable = entropies < self.entropy_margin
        if not reliable.any():
            return
        entropies[reliable].mean().backward()
        self.optimizer.first_step()
        perturbed_entropies = entropy(self.model(images))[reliable]
        second_reliable = perturbed_entropies < self.entropy_margin
        if not second_reliable.any():
            self.optimizer.second_step(update=False)
            return
        second_loss = perturbed_entropies[second_reliable].mean()
        second_loss.backward()
        self.optimizer.second_step(update=True)
        value = float(second_loss.detach())
        self.sar_entropy_ema = value if self.sar_entropy_ema is None else 0.9 * self.sar_entropy_ema + 0.1 * value
        if self.sar_entropy_ema < self.sar_reset_constant:
            self.reset_representation()
            self.enable_representation()

    @staticmethod
    def patch_shuffle(images: torch.Tensor, grid: int = 4) -> torch.Tensor:
        batch, channels, height, width = images.shape
        if height % grid or width % grid:
            raise ValueError("DeYO patch shuffle requires spatial dimensions divisible by its grid")
        patch_height, patch_width = height // grid, width // grid
        tiles = (
            images.view(batch, channels, grid, patch_height, grid, patch_width)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, grid * grid, channels, patch_height, patch_width)
        )
        permutation = torch.rand(batch, grid * grid, device=images.device).argsort(dim=1)
        gather_index = permutation[:, :, None, None, None].expand_as(tiles)
        shuffled = tiles.gather(1, gather_index)
        return (
            shuffled.view(batch, grid, grid, channels, patch_height, patch_width)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, height, width)
        )

    def adapt_deyo(self, images: torch.Tensor) -> None:
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        entropies = entropy(logits)
        first_filter = entropies < 0.5 * float(np.log(logits.shape[1]))
        if not first_filter.any():
            return
        selected_images = self.patch_shuffle(images[first_filter].detach())
        with torch.no_grad():
            shuffled_logits = self.model(selected_images)
        selected_logits = logits[first_filter]
        selected_entropies = entropies[first_filter]
        predicted_class = selected_logits.argmax(1, keepdim=True)
        original_probability = selected_logits.softmax(1).gather(1, predicted_class).squeeze(1)
        shuffled_probability = shuffled_logits.softmax(1).gather(1, predicted_class).squeeze(1)
        plpd = original_probability - shuffled_probability
        second_filter = plpd > 0.2
        if not second_filter.any():
            return
        retained_entropy = selected_entropies[second_filter]
        retained_plpd = plpd[second_filter]
        weights = torch.exp(self.entropy_margin - retained_entropy.detach()) + torch.exp(retained_plpd.detach())
        (retained_entropy * weights).mean().backward()
        self.optimizer.step()

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        if self.adaptation_kind == "ods" and self.action in (
            Action.REPRESENTATION_ONLY,
            Action.JOINT,
        ):
            logits, labels = self.ods_outputs(images)
            self.ods_pending = (images.detach().cpu(), labels.detach().cpu())
            if self.action == Action.JOINT:
                logits = prior_corrected_logits(
                    logits,
                    self.target_prior,
                    self.source_prior,
                    strength=self.prior_config.strength,
                    ratio_clip=self.prior_config.ratio_clip,
                )
            return logits
        self.model.eval()
        logits = self.model(images)
        if self.action in (Action.PRIOR_ONLY, Action.JOINT):
            logits = prior_corrected_logits(
                logits,
                self.target_prior,
                self.source_prior,
                strength=self.prior_config.strength,
                ratio_clip=self.prior_config.ratio_clip,
            )
        return logits

    def transition(self, action: Action, diagnosis, images: torch.Tensor) -> float:
        started = time.perf_counter()
        previously_representation = self.action in (
            Action.REPRESENTATION_ONLY,
            Action.JOINT,
        )
        next_representation = action in (Action.REPRESENTATION_ONLY, Action.JOINT)
        if not next_representation and previously_representation:
            self.reset_representation()
        elif next_representation and not previously_representation:
            self.enable_representation()
        if action in (Action.FREEZE, Action.REPRESENTATION_ONLY):
            self.target_prior = self.source_prior.clone()
        if action in (Action.PRIOR_ONLY, Action.JOINT):
            estimate = diagnosis.target_prior.detach()
            self.target_prior = (
                self.prior_config.momentum * self.target_prior
                + (1.0 - self.prior_config.momentum) * estimate
            )
            self.target_prior = self.target_prior / self.target_prior.sum()
        if action in (Action.REPRESENTATION_ONLY, Action.JOINT):
            if self.optimizer is None:
                raise RuntimeError("representation action selected without an adaptation optimizer")
            if self.adaptation_kind == "eta":
                self.adapt_eta(images)
            elif self.adaptation_kind == "sar":
                self.adapt_sar(images)
            elif self.adaptation_kind == "deyo":
                self.adapt_deyo(images)
            elif self.adaptation_kind == "ods":
                self.adapt_ods(images)
            else:
                self.model.eval()
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                if action == Action.JOINT:
                    logits = prior_corrected_logits(
                        logits,
                        self.target_prior,
                        self.source_prior,
                        strength=self.prior_config.strength,
                        ratio_clip=self.prior_config.ratio_clip,
                    )
                loss = entropy(logits).mean()
                loss.backward()
                self.optimizer.step()
        self.action = action
        return time.perf_counter() - started


def normalized_images(array: np.ndarray, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = np.asarray(array[indices.cpu().numpy()]).copy()
    tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2).div_(255.0)
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616), device=device).view(1, 3, 1, 1)
    return ((tensor - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def validate_cached_raw_logits(
    model: nn.Module,
    raw_banks: Dict[str, np.ndarray],
    feature_banks: Dict[str, FeatureBank],
    device: torch.device,
    samples: int = 64,
    maximum_tolerance: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    audit: Dict[str, Dict[str, float]] = {}
    for name, feature_bank in feature_banks.items():
        count = min(samples, len(feature_bank.labels))
        indices = torch.arange(count, dtype=torch.long)
        images = normalized_images(raw_banks[name], indices, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            direct = model(images).float().cpu()
        cached = feature_bank.logits[indices].float().cpu()
        absolute = (direct - cached).abs()
        record = {
            "samples": count,
            "max_absolute_logit_error": float(absolute.max()),
            "mean_absolute_logit_error": float(absolute.mean()),
            "top1_agreement": float((direct.argmax(1) == cached.argmax(1)).float().mean()),
            "maximum_tolerance": maximum_tolerance,
        }
        audit[name] = record
        if record["max_absolute_logit_error"] > maximum_tolerance:
            raise RuntimeError(f"cached/raw logit mismatch for {name}: {record}")
    return audit


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_artifact = torch.load(args.source, map_location="cpu", weights_only=False)
    development_artifact = torch.load(args.development, map_location="cpu", weights_only=False)
    calibration_artifact = torch.load(args.calibration, map_location="cpu", weights_only=False)
    clean, clean_meta = load_bank(args.clean, "clean")
    representation_a, meta_a = load_bank(args.representation_a, "representation_a")
    representation_b, meta_b = load_bank(args.representation_b, "representation_b")
    if not torch.equal(clean.labels, representation_a.labels) or not torch.equal(
        clean.labels, representation_b.labels
    ):
        raise ValueError("cached raw-image conditions need identical ordered labels")

    classes = clean.classes
    uniform = torch.full((classes,), 1.0 / classes)
    forward = long_tailed_prior(classes, args.imbalance_ratio)
    backward = long_tailed_prior(classes, args.imbalance_ratio, reverse=True)
    phases = [
        StreamPhase("clean_start", "clean", args.phase_windows, Action.FREEZE, uniform),
        StreamPhase("prior_forward", "clean", args.phase_windows, Action.PRIOR_ONLY, forward),
        StreamPhase("clean_recovery", "clean", args.phase_windows, Action.FREEZE, uniform),
        StreamPhase("representation_a", "representation_a", args.phase_windows, Action.REPRESENTATION_ONLY, uniform),
        StreamPhase("compound_forward", "representation_a", args.phase_windows, Action.JOINT, forward),
        StreamPhase("representation_b", "representation_b", args.phase_windows, Action.REPRESENTATION_ONLY, uniform),
        StreamPhase("compound_backward", "representation_b", args.phase_windows, Action.JOINT, backward),
        StreamPhase("clean_end", "clean", args.phase_windows, Action.FREEZE, uniform),
    ]
    windows = build_stream(
        {"clean": clean, "representation_a": representation_a, "representation_b": representation_b},
        phases,
        window_size=args.window_size,
        seed=args.seed,
    )

    source_statistics = fit_source_statistics(
        source_artifact["features"].float().to(device),
        source_artifact["labels"].long().to(device),
    )
    prior_scale, representation_scale = development_scales_from_clean_windows(
        development_artifact["features"].float(),
        development_artifact["labels"].long(),
        source_statistics,
        window_size=args.window_size,
        windows=300,
        quantile=0.90,
        seed=args.seed,
        device=device,
    )
    prior_reliability_threshold = development_prior_reliability_from_clean_windows(
        development_artifact["features"].float(),
        development_artifact["labels"].long(),
        source_statistics,
        window_size=args.window_size,
        windows=300,
        lower_quantile=0.05,
        seed=args.seed,
        device=device,
    )
    thresholds = calibrate_from_clean_windows(
        calibration_artifact["features"].float(),
        calibration_artifact["labels"].long(),
        source_statistics,
        window_size=args.window_size,
        windows=None,
        familywise_alpha=0.05,
        seed=args.seed,
        device=device,
        sampling="disjoint",
        prior_scale=prior_scale,
        representation_scale=representation_scale,
    )
    router = HysteresisRouter(thresholds.to_router_config())

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_model = build_model(int(checkpoint["classes"]))
    source_model.load_state_dict(checkpoint["model"])
    source_model.eval().to(device).to(memory_format=torch.channels_last)
    source_state = {key: value.detach().cpu().clone() for key, value in source_model.state_dict().items()}
    prior_config = PriorAdapterConfig(
        strength=args.prior_strength,
        momentum=args.prior_momentum,
        ratio_clip=args.prior_ratio_clip,
    )
    if args.method_set == "ods_router":
        names = ("source", "shiftroute_ods")
    else:
        names = (
            "source",
            "always_prior",
            "prequential_tent",
            "prequential_eta",
            "prequential_sar",
            "prequential_deyo",
            "prequential_ods",
            "prequential_joint",
            "shiftroute_tent",
            "oracle_tent",
        )
    states = {
        name: RawState(
            name,
            source_model,
            source_state,
            source_statistics.source_prior,
            classes,
            args.tent_learning_rate,
            args.eta_entropy_margin,
            args.eta_diversity_margin if args.eta_diversity_margin is not None else (0.4 if args.dataset == "cifar10" else 0.1),
            args.sar_reset_constant,
            args.ods_learning_rate,
            args.ods_queue_size,
            args.ods_knn,
            args.seed,
            prior_config,
        )
        for name in names
    }

    data_type = dataset_class(args.dataset)
    clean_dataset = data_type(str(args.data_root), train=False, download=False)
    clean_original_indices = clean_meta.get("indices", torch.arange(len(clean.labels))).long().numpy()
    a_original_indices = meta_a.get("indices", torch.arange(len(representation_a.labels))).long().numpy()
    b_original_indices = meta_b.get("indices", torch.arange(len(representation_b.labels))).long().numpy()
    raw_a = np.load(
        args.corruption_root / f"{meta_a['corruption']}.npy", mmap_mode="r"
    )[(int(meta_a["severity"]) - 1) * 10000 : int(meta_a["severity"]) * 10000]
    raw_b = np.load(
        args.corruption_root / f"{meta_b['corruption']}.npy", mmap_mode="r"
    )[(int(meta_b["severity"]) - 1) * 10000 : int(meta_b["severity"]) * 10000]
    raw_banks = {
        "clean": clean_dataset.data[clean_original_indices],
        "representation_a": raw_a[a_original_indices],
        "representation_b": raw_b[b_original_indices],
    }
    cache_raw_consistency = validate_cached_raw_logits(
        source_model,
        raw_banks,
        {"clean": clean, "representation_a": representation_a, "representation_b": representation_b},
        device,
    )

    rows = []
    torch.cuda.reset_peak_memory_stats()
    for window in windows:
        if window.sample_indices is None:
            raise RuntimeError("stream did not retain raw sample indices")
        images = normalized_images(raw_banks[window.bank_name], window.sample_indices, device)
        labels = window.labels.to(device)
        diagnosis = diagnose_window(window.features.to(device), source_statistics)
        routed_action = router.update(
            float(diagnosis.prior_score),
            float(diagnosis.representation_score),
            confidence=float(diagnosis.mean_assignment_confidence),
        )
        routed_action, prior_reliable = apply_prior_reliability_guard(
            routed_action,
            assignment_confidence=float(diagnosis.mean_assignment_confidence),
            threshold=prior_reliability_threshold,
        )
        for name, state in states.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = state.predict(images)
            torch.cuda.synchronize()
            prediction_seconds = time.perf_counter() - started
            predictions = logits.argmax(1)
            next_action = selected_action(name, window.true_action, routed_action)
            rows.append(
                {
                    "window": window.window_index,
                    "phase": window.phase_name,
                    "method": name,
                    "true_action": window.true_action.value,
                    "action_applied": state.action.value,
                    "action_selected_next": next_action.value,
                    "accuracy": float(predictions.eq(labels).float().mean()),
                    "nll": float(F.cross_entropy(logits, labels)),
                    "ece": expected_calibration_error(logits, labels),
                    "prior_score": float(diagnosis.prior_score),
                    "representation_score": float(diagnosis.representation_score),
                    "assignment_confidence": float(diagnosis.mean_assignment_confidence),
                    "prior_reliable": bool(prior_reliable),
                    "prediction_seconds": prediction_seconds,
                }
            )
        for name, state in states.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            state.transition(
                selected_action(name, window.true_action, routed_action), diagnosis, images
            )
            torch.cuda.synchronize()
            rows[-len(states) + names.index(name)]["update_seconds"] = (
                time.perf_counter() - started
            )

    frame = pd.DataFrame(rows)
    source_accuracy = frame[frame.method == "source"].set_index("window").accuracy
    frame["source_accuracy"] = frame.window.map(source_accuracy)
    frame["gain_over_source"] = frame.accuracy - frame.source_accuracy
    if args.method_set == "ods_router":
        # The frozen source is evaluated internally to construct paired gains,
        # but omitted from this auxiliary artifact so aggregation with the full
        # suite cannot duplicate source observations.
        frame = frame[frame.method == "shiftroute_ods"].reset_index(drop=True)
    window_metrics_path = args.output_dir / "window_metrics.csv"
    frame.to_csv(window_metrics_path, index=False)
    summary = (
        frame.groupby("method")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            worst_window_accuracy=("accuracy", "min"),
            mean_nll=("nll", "mean"),
            mean_ece=("ece", "mean"),
            mean_gain=("gain_over_source", "mean"),
            negative_transfer_rate_0p5=("gain_over_source", lambda value: float((value < -0.005).mean())),
            severe_negative_transfer_rate=("gain_over_source", lambda value: float((value < -0.02).mean())),
            mean_prediction_seconds=("prediction_seconds", "mean"),
            mean_update_seconds=("update_seconds", "mean"),
        )
        .reset_index()
    )
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "thresholds": thresholds.__dict__,
        "input_sha256": {
            "checkpoint": sha256(args.checkpoint),
            "source": sha256(args.source),
            "development": sha256(args.development),
            "calibration": sha256(args.calibration),
            "clean": sha256(args.clean),
            "representation_a": sha256(args.representation_a),
            "representation_b": sha256(args.representation_b),
        },
        "calibration_protocol": {
            "mode": "disjoint_max",
            "identity_disjoint_windows": True,
            "effective_windows": thresholds.calibration_windows,
            "familywise_alpha": 0.05,
            "development_scale_quantile": 0.90,
            "prior_reliability_lower_quantile": 0.05,
            "prior_reliability_threshold": prior_reliability_threshold,
        },
        "output_sha256": {
            "window_metrics": sha256(window_metrics_path),
            "summary": sha256(summary_path),
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "protocol": "predict_current_window_then_unlabeled_update_for_next_window",
        "tent_variant": "causal_prequential_bn_affine_entropy_update",
        "external_baseline_protocol": {
            "eta": "EATA without Fisher regularization; entropy and diversity filtering follow the public EATA implementation",
            "sar": "reliable entropy filtering plus first-order SAM and entropy-EMA recovery follow the public SAR implementation",
            "deyo": "entropy and PLPD filtering with per-image patch shuffling follow the public DeYO implementation",
            "ods": "official defaults: learned BN statistics (momentum 0.01), LAME kNN refinement, geometric-mean predictions, pseudo-label-balanced reservoir, and class-frequency-weighted entropy updates",
            "ods_reference_implementation": "https://www.lamda.nju.edu.cn/code_ODS.ashx",
            "ods_causal_alignment": "current-window prediction is logged before its unlabeled reservoir/entropy update; the update affects only the next window",
            "reference_implementation": "https://github.com/mariodoebler/test-time-adaptation",
            "reference_commit": "8b2ed63255d906b6b0afaefc6ae630b9ec4bdaa1",
        },
        "cache_raw_consistency": cache_raw_consistency,
        "labels_used_for_controller_or_update": False,
        "auxiliary_method_only": args.method_set == "ods_router",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
