from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

logger = logging.getLogger("size_tokenizer")

MEDIAN_CANDIDATE_FLOOR = 4
SINGLETON_CEILING = 0.15
PROXY_MEDIAN_CALIBRATION = 1.5
PROXY_MEDIAN_FLOOR = MEDIAN_CANDIDATE_FLOOR * PROXY_MEDIAN_CALIBRATION
POIS_PER_CODE_FLOOR = 1.1
POIS_PER_CODE_TARGET = (1.2, 1.3)


@dataclass(frozen=True)
class TokenizerSizingCandidate:
    codebook_size: int
    depth3_median_candidates: float
    depth3_singleton_rate: float
    passes_depth3_gate: bool
    n_codebooks: int | None
    pois_per_code: float | None
    in_target_window: bool


@dataclass(frozen=True)
class TokenizerSizingReport:
    poi_count: int
    input_dim: int
    text_dim: int
    graph_dim: int
    text_graph_scale: float
    candidates: tuple[TokenizerSizingCandidate, ...]


def _load_aligned(places: Path, text_dir: Path, graph_dir: Path, pca_dim: int):
    pi = pl.read_parquet(places)
    gi = pl.read_parquet(graph_dir / "places_index.parquet")
    graph = (
        torch.load(graph_dir / "poi_graph_emb.pt", map_location="cpu")
        .float()
        .numpy()
    )
    ti = pl.read_parquet(text_dir / "text_embeddings.index.parquet")
    text = np.load(text_dir / "text_embeddings.npy", mmap_mode="r")

    def _key(*frames: pl.DataFrame) -> str:
        for candidate in ("placekey", "placekey_id"):
            if all(candidate in frame.columns for frame in frames):
                return candidate
        raise ValueError(
            "no shared join key across places/graph/text indices; have "
            + "; ".join(str(frame.columns) for frame in frames)
        )

    graph_key = _key(pi, gi)
    text_key = _key(pi, ti)
    joined = pi.join(
        gi.select(graph_key).with_columns(
            pl.Series("g", np.arange(graph.shape[0]))
        ),
        on=graph_key,
        maintain_order="left",
    ).join(
        ti.select(text_key).with_columns(
            pl.Series("t", ti["embedding_row"].to_numpy())
        ),
        on=text_key,
        maintain_order="left",
    )
    if joined.height != pi.height:
        raise ValueError(
            f"join on {graph_key}/{text_key} kept {joined.height} of "
            f"{pi.height} POIs; the artifacts are not aligned"
        )
    g = graph[joined["g"].to_numpy()]
    t = np.asarray(text[joined["t"].to_numpy()], dtype=np.float32)
    centred = t - t.mean(axis=0, keepdims=True)
    sample = np.random.default_rng(0).choice(
        centred.shape[0], min(30000, centred.shape[0]), replace=False
    )
    _, _, basis = np.linalg.svd(centred[sample], full_matrices=False)
    return np.hstack([centred @ basis[:pca_dim].T, g]).astype(
        np.float32
    ), joined.height


def _standardize(x: np.ndarray, n_text: int, scale: float) -> np.ndarray:
    out = (x - x.mean(0, keepdims=True)) / x.std(0, keepdims=True).clip(1e-8)
    out[:, :n_text] *= scale
    return out


def _residual_kmeans(
    x: np.ndarray, k: int, levels: int, device: str, iters: int = 25
):
    data = torch.tensor(x, device=device)
    codes = np.zeros((data.shape[0], levels), dtype=np.int64)
    for level in range(levels):
        generator = torch.Generator(device=device).manual_seed(level)
        centres = data[
            torch.randperm(data.shape[0], generator=generator, device=device)[
                :k
            ]
        ].clone()
        for _ in range(iters):
            assigned = _assign(data, centres)
            summed = torch.zeros_like(centres)
            counts = torch.zeros(k, device=device)
            summed.index_add_(0, assigned, data)
            counts.index_add_(
                0, assigned, torch.ones(data.shape[0], device=device)
            )
            centres = torch.where(
                (counts == 0)[:, None],
                centres,
                summed / counts.clamp_min(1)[:, None],
            )
        assigned = _assign(data, centres)
        codes[:, level] = assigned.cpu().numpy()
        data = data - centres[assigned]
    return codes


def _assign(data: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    out = torch.empty(data.shape[0], dtype=torch.long, device=data.device)
    norms = (centres * centres).sum(1)
    for start in range(0, data.shape[0], 65536):
        chunk = data[start : start + 65536]
        out[start : start + 65536] = (
            norms[None, :] - 2.0 * (chunk @ centres.T)
        ).argmin(1)
    return out


def _profile(codes: np.ndarray, k: int, depth: int):
    keys = np.zeros(codes.shape[0], dtype=np.int64)
    for level in range(depth):
        keys = keys * k + codes[:, level]
    _, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    per_poi = counts[inverse]
    return (
        len(counts),
        float(np.median(per_poi)),
        float((per_poi == 1).mean()),
        codes.shape[0] / len(counts),
    )


def size_tokenizer(
    *,
    places: Path,
    text_dir: Path,
    graph_dir: Path,
    codebook_sizes: list[int],
    max_levels: int = 8,
    pca_text_dim: int = 16,
    text_graph_scale: float = 4.0,
    device: str = "cuda:0",
) -> TokenizerSizingReport:
    if not codebook_sizes or any(size < 2 for size in codebook_sizes):
        raise ValueError(
            "codebook_sizes must contain integers greater than one"
        )
    if max_levels < 3:
        raise ValueError("max_levels must be at least three")
    if pca_text_dim < 1:
        raise ValueError("pca_text_dim must be positive")
    raw, n_pois = _load_aligned(places, text_dir, graph_dir, pca_text_dim)
    x = _standardize(raw, pca_text_dim, text_graph_scale)
    logger.info(
        "%s POIs  input %dd (%d text + %d graph, text scaled x%g)",
        f"{n_pois:,}",
        x.shape[1],
        pca_text_dim,
        x.shape[1] - pca_text_dim,
        text_graph_scale,
    )
    logger.info(
        "gates: depth-3 median >= %d and singleton <= %.0f%%; "
        "full SID at or above %.1f POIs/code",
        PROXY_MEDIAN_FLOOR,
        100 * SINGLETON_CEILING,
        POIS_PER_CODE_FLOOR,
    )
    candidates: list[TokenizerSizingCandidate] = []
    for k in codebook_sizes:
        codes = _residual_kmeans(x, k, max_levels, device)
        _, median, singleton, _ = _profile(codes, k, 3)
        passes = (
            median >= PROXY_MEDIAN_FLOOR and singleton <= SINGLETON_CEILING
        )
        ratios = {
            d: _profile(codes, k, d)[3] for d in range(3, max_levels + 1)
        }
        above = [d for d, r in ratios.items() if r >= POIS_PER_CODE_FLOOR]
        depth = max(above) if above else None
        in_window = (
            depth is not None
            and POIS_PER_CODE_TARGET[0]
            <= ratios[depth]
            <= POIS_PER_CODE_TARGET[1]
        )
        logger.info(
            "  k=%-3d depth-3 median %2.0f  singleton %5.1f%%  %s | "
            "n_codebooks -> %s (%s POIs/code%s)",
            k,
            median,
            100 * singleton,
            "PASS" if passes else "fail",
            depth if depth else "NONE (never reaches the floor)",
            f"{ratios[depth]:.2f}" if depth else "--",
            "" if in_window else ", outside the 1.2-1.3 window",
        )
        candidates.append(
            TokenizerSizingCandidate(
                codebook_size=k,
                depth3_median_candidates=median,
                depth3_singleton_rate=singleton,
                passes_depth3_gate=passes,
                n_codebooks=depth,
                pois_per_code=ratios[depth] if depth is not None else None,
                in_target_window=in_window,
            )
        )
    logger.info(
        "Calibration: the PASS flag above already applies the %gx proxy "
        "correction, so it is testing a proxy median of %g against the "
        "fitted median-%d gate. Confirm the chosen configuration with one "
        "tokenizer run; the proxy chooses K, it does not replace a fit.",
        PROXY_MEDIAN_CALIBRATION,
        PROXY_MEDIAN_FLOOR,
        MEDIAN_CANDIDATE_FLOOR,
    )
    return TokenizerSizingReport(
        poi_count=n_pois,
        input_dim=x.shape[1],
        text_dim=pca_text_dim,
        graph_dim=x.shape[1] - pca_text_dim,
        text_graph_scale=text_graph_scale,
        candidates=tuple(candidates),
    )


__all__ = [
    "TokenizerSizingCandidate",
    "TokenizerSizingReport",
    "size_tokenizer",
]
