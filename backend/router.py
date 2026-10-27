"""AHP-weighted, interpretable retrieval strategy selection."""
from __future__ import annotations
from typing import Any

# Criteria order: graph intent, graph coverage, vector relevance, cross-document need,
# latency/cost suitability. Saaty pairwise comparison matrix (configurable in code).
AHP_MATRIX = [
    [1,   2,   2,   1/2, 2],
    [1/2, 1,   1,   1/3, 2],
    [1/2, 1,   1,   1/2, 2],
    [2,   3,   2,   1,   3],
    [1/2, 1/2, 1/2, 1/3, 1],
]
CRITERIA = ["graph_intent", "graph_coverage", "vector_relevance", "cross_document_need", "latency_cost_suitability"]


def ahp_weights(matrix: list[list[float]] = AHP_MATRIX) -> tuple[list[float], float]:
    """Compute principal-eigenvector approximation via geometric means and consistency ratio."""
    n = len(matrix)
    gm = []
    for row in matrix:
        product = 1.0
        for value in row:
            product *= value
        gm.append(product ** (1.0 / n))
    total = sum(gm) or 1.0
    weights = [x / total for x in gm]
    weighted_sum = [sum(matrix[i][j] * weights[j] for j in range(n)) for i in range(n)]
    lambda_max = sum(weighted_sum[i] / weights[i] for i in range(n)) / n
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    cr = ci / ri_table.get(n, 1.49) if ri_table.get(n, 0) else 0.0
    return weights, cr


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def route_query(query: str, graph_coverage: float, vector_relevance: float, storage: dict[str, Any], latency_sensitive: bool = False) -> dict[str, Any]:
    q = query.lower()
    graph_terms = ("relationship", "connected", "connection", "related to", "path between", "who works", "depends on", "linked to", "relationship between", "graph", "how is")
    broad_terms = ("compare", "across documents", "all documents", "summarize", "overview", "different documents", "multiple documents", "all files", "contrast")
    graph_intent = 0.85 if any(term in q for term in graph_terms) else 0.45
    cross_document = 0.9 if any(term in q for term in broad_terms) else 0.35
    graph_coverage = _clamp(graph_coverage)
    vector_relevance = _clamp(vector_relevance)
    latency_suitability = 0.8 if latency_sensitive else 0.5
    metrics = {
        "graph_intent": graph_intent,
        "graph_coverage": graph_coverage,
        "vector_relevance": vector_relevance,
        "cross_document_need": cross_document,
        "latency_cost_suitability": latency_suitability,
    }
    weights, consistency_ratio = ahp_weights()
    # Suitability for Way 1: graph-first and cheaper when graph is likely sufficient.
    way1_metrics = [graph_intent, graph_coverage, 1.0 - vector_relevance * 0.45, 1.0 - cross_document, latency_suitability]
    # Way 2: parallel retrieval rewards vector relevance and multi-document coverage.
    way2_metrics = [1.0 - graph_intent * 0.35, 0.55 + graph_coverage * 0.45, vector_relevance, cross_document, 1.0 - latency_suitability * 0.35]
    way1_score = sum(w * v for w, v in zip(weights, way1_metrics))
    way2_score = sum(w * v for w, v in zip(weights, way2_metrics))
    selected = "way2" if way2_score >= way1_score else "way1"
    reasons = []
    if cross_document >= 0.7:
        reasons.append("The query appears to require evidence across multiple documents, favoring parallel retrieval.")
    if graph_intent >= 0.7 and graph_coverage >= 0.6:
        reasons.append("The query is relationship-oriented and the graph appears to have useful coverage.")
    if vector_relevance >= 0.65:
        reasons.append("Initial vector similarity indicates relevant semantic evidence is available in Qdrant.")
    if graph_coverage < 0.35:
        reasons.append("Estimated graph coverage is low, so vector evidence should be considered early.")
    if not reasons:
        reasons.append("The weighted scores compare graph intent, graph coverage, vector relevance, document breadth, and latency/cost preferences.")
    return {
        "selected_strategy": selected,
        "scores": {"way1": round(way1_score, 4), "way2": round(way2_score, 4)},
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "weights": {k: round(v, 4) for k, v in zip(CRITERIA, weights)},
        "ahp_consistency_ratio": round(consistency_ratio, 4),
        "ahp_consistency_acceptable": consistency_ratio < 0.1,
        "storage": storage,
        "reasons": reasons,
        "explanation": f"Selected {selected.upper()} because its weighted suitability score was higher ({way1_score:.3f} vs {way2_score:.3f}). These are routing scores, not probabilities of answer correctness."
    }
