"""
RRF (Reciprocal Rank Fusion) -- combines a sparse (BM25) ranking and a dense
(embedding) ranking into one fused ranking, using ranks rather than raw
scores (which live on incomparable scales). Matches the weekly update's
pseudocode: score = 1/(k + rank_sparse) + 1/(k + rank_dense), k=60 default.
"""


def _ranks_from_scored_list(scored: list[tuple[int, float]]) -> dict[int, int]:
    """scored is already sorted descending by score; rank 1 = best."""
    return {pid: i + 1 for i, (pid, _score) in enumerate(scored)}


def rrf_fuse(
    sparse_ranked: list[tuple[int, float]],
    dense_ranked: list[tuple[int, float]],
    k: int = 60,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[tuple[int, float]]:
    """Returns [(provider_id, rrf_score), ...] sorted descending."""
    sparse_ranks = _ranks_from_scored_list(sparse_ranked)
    dense_ranks = _ranks_from_scored_list(dense_ranked)
    all_ids = set(sparse_ranks) | set(dense_ranks)

    fused = []
    for pid in all_ids:
        score = 0.0
        if pid in sparse_ranks:
            score += sparse_weight * (1.0 / (k + sparse_ranks[pid]))
        if pid in dense_ranks:
            score += dense_weight * (1.0 / (k + dense_ranks[pid]))
        fused.append((pid, score))

    return sorted(fused, key=lambda x: x[1], reverse=True)


def rrf_fuse_many(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """rrf_fuse for any number of rankings (e.g. BM25 + dense + taxonomy). A ranking
    may be partial -- the taxonomy tower only returns providers sharing a skill --
    and a provider missing from it just gets no contribution from that list."""
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[int, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for pid, rank in _ranks_from_scored_list(ranked).items():
            fused[pid] = fused.get(pid, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
