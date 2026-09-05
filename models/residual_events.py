"""RGB-only foreground residual codes and factorized-control warm starts."""

import cv2
import numpy as np
import torch
from models.motion_codes import assign_codes


def residual_features(previous, following, motion_ids, flow_centers):
    """Hand-selected foreground ROI; no semantic labels or engine measurements."""
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    rows = []
    for before, after, motion in zip(previous, following, motion_ids):
        flow = cv2.resize(flow_centers[motion].reshape(4, 4, 2), (64, 64))
        predicted = cv2.remap(
            before.astype(np.float32) / 127.5 - 1,
            xx - flow[..., 0],
            yy - flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        difference = after.astype(np.float32) / 127.5 - 1 - predicted
        rows.append(
            cv2.resize(
                difference[32:58, 20:44], (8, 8), interpolation=cv2.INTER_AREA
            ).reshape(-1)
        )
    return np.asarray(rows, dtype=np.float32)


def infer_events(previous, following, motion_ids, bank):
    features = residual_features(previous, following, motion_ids, bank["flow_centers"])
    latent = (features - bank["event_pca_mean"]) @ bank["event_pca_components"].T
    return assign_codes(latent, bank["event_centers"])


def expand_motion_state(state, history, event_count):
    """Add initially neutral event features without changing motion predictions.

    Optimizer state is handled separately: a warm start must explicitly reset or
    migrate moments for the expanded input layer, never silently load wrong shapes.
    """
    expanded = dict(state)
    old = state["actions.0.weight"]
    motion_count = old.shape[1] // history
    weights = old.new_zeros(old.shape[0], history, motion_count + event_count)
    weights[:, :, :motion_count] = old.reshape(old.shape[0], history, motion_count)
    expanded["actions.0.weight"] = weights.flatten(1)
    if "motion_centers" in state:
        expanded["motion_centers"] = state["motion_centers"].repeat_interleave(
            event_count, 0
        )
    return expanded


def make_event_prior(bank):
    """Decode train-fitted residual prototypes relative to the modal event.

    This deliberately imposes a learned mean visual effect. Its presence alone
    is not evidence of learned firing mechanics; compare generation against the
    prototype-only baseline and shuffled events on held-out RGB.
    """
    vectors = (
        bank["event_pca_mean"] + bank["event_centers"] @ bank["event_pca_components"]
    )
    vectors = (vectors - vectors[:1]).reshape(-1, 8, 8, 3)
    prior = np.zeros((len(vectors), 3, 64, 64), dtype=np.float32)

    def taper(n):
        return np.minimum(1, np.minimum(np.arange(n) + 1, n - np.arange(n)) / 3)

    mask = (taper(26)[:, None] * taper(24)[None, :]).astype(np.float32)
    for i, field in enumerate(vectors):
        dense = (
            cv2.resize(field, (24, 26), interpolation=cv2.INTER_LINEAR)
            * mask[..., None]
        )
        prior[i, :, 32:58, 20:44] = dense.transpose(2, 0, 1)
    return prior
