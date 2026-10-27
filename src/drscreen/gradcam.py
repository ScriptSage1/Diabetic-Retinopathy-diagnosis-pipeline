"""Grad-CAM written by hand with forward/backward hooks (Selvaraju et al., 2017).

A Grad-CAM map shows which image regions most increased the chosen output score for THIS model. It is a
model-attention visualisation, not evidence that a lesion is present or that the model reasons like a clinician.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np
import torch
import torch.nn as nn

from .models import gradcam_target_layer
from .preprocess import circle_mask


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        self.layer = target_layer if target_layer is not None else gradcam_target_layer(model)
        self._act = None
        self._grad = None
        self._h1 = self.layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self._act = out.detach()
        out.register_hook(lambda g: setattr(self, "_grad", g.detach()))

    def remove(self):
        self._h1.remove()

    def __call__(self, x: torch.Tensor, target="pred"):
        """x: (1,3,H,W) float tensor on the model's device. target: 'pred' (predicted grade), 'referable'
        (P(grade>=2)), or an int grade. Returns (cam HxW float in [0,1], logits ndarray)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(False)
            logits = self.model(x)
            if target == "referable":
                score = torch.softmax(logits, 1)[:, 2:].sum()
            else:
                k = int(logits.argmax(1)) if target == "pred" else int(target)
                score = logits[0, k]
            score.backward()
        w = self._grad.mean(dim=(2, 3), keepdim=True)  # channel weights = spatially averaged gradients
        cam = torch.relu((w * self._act).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = cam * circle_mask(cam.shape[0])  # confine to the field of view
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32), logits.detach().float().cpu().numpy()[0]


def overlay_heatmap(rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET heatmap onto an RGB uint8 image of the same size; no colour where the map is ~0."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    a = (alpha * np.clip(cam * 2.0, 0, 1))[..., None]  # fade weak activations
    return np.clip((1 - a) * rgb_uint8 + a * heat, 0, 255).astype(np.uint8)


def randomize_weights(model: nn.Module, seed: int = 0) -> nn.Module:
    """Return a copy of the model with every layer re-initialised by its own default initialiser
    (BatchNorm scales become 1, so activations stay alive), for the weight-randomisation sanity check."""
    m = copy.deepcopy(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for mod in m.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.reset_running_stats()
    return m


def randomization_sanity_check(model: nn.Module, x: torch.Tensor, target="pred") -> dict:
    """Adebayo et al. (2018) style check: a map that does not change when the weights are randomised is not
    explaining the model. Returns the Spearman correlation between trained and randomised maps (lower is better).
    This is an indicative check on one image, not a proof."""
    from scipy.stats import spearmanr

    g1 = GradCAM(model)
    cam_t, _ = g1(x, target)
    g1.remove()
    rnd = randomize_weights(model).to(x.device)
    g2 = GradCAM(rnd)
    cam_r, _ = g2(x, target)
    g2.remove()
    mask = circle_mask(cam_t.shape[0])
    if cam_t[mask].std() == 0 or cam_r[mask].std() == 0:
        return {"spearman": float("nan"), "note": "one of the maps is constant"}
    rho = float(spearmanr(cam_t[mask].ravel(), cam_r[mask].ravel())[0])
    return {"spearman": rho, "note": "trained vs weight-randomised map; values near 0 indicate the map depends on learned weights"}
