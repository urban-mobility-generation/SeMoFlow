from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semoflow.metrics.tokenization import (
        SemanticIDEvaluation,
    )


def _format_metric(
    values: Mapping[Any, Any] | None,
    depth: Any,
) -> str:
    if values is None or depth not in values:
        return "n/a"
    value = values[depth]
    return "n/a" if value is None else f"{value:.4f}"


def _format_ci(values: dict[int, list[float]] | None, depth: int) -> str:
    if values is None or depth not in values:
        return "n/a"
    interval = values[depth]
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def _format_count_metric(values: Mapping[int, int] | None, depth: int) -> str:
    if values is None or depth not in values:
        return "n/a"
    return f"{values[depth]:,}"


def write_semantic_id_eval_report(
    result: SemanticIDEvaluation,
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Semantic ID Evaluation Report\n")

    lines.append("## SID Quality\n")
    lines.append(f"- POIs: {result.n_pois:,}")
    lines.append(f"- Unique code tuples: {result.n_keys:,}")
    lines.append(f"- Codebooks: {result.n_codebooks}")
    lines.append(f"- Codebook size: {result.codebook_size}")
    lines.append(f"- Collision rate: {result.collision_rate:.4f}")
    lines.append(f"- Ambiguous key rate: {result.ambiguous_key_rate:.4f}")
    lines.append(f"- Singleton key rate: {result.singleton_key_rate:.4f}")
    if result.category_purity_mean is not None:
        lines.append(
            "- Full-SID macro category purity "
            "(singleton-dominated diagnostic): "
            f"{result.category_purity_mean:.4f}"
        )
    lines.append("")

    lines.append("### Codebook Utilization\n")
    lines.append(
        "| Level | Utilization | Active | Perplexity | Gini | Top-1 |"
    )
    lines.append(
        "|-------|-------------|--------|------------|------|-------|"
    )
    for i, util in enumerate(result.codebook_utilization):
        active = result.codebook_active_codes[i]
        perplexity = result.codebook_perplexity[i]
        gini = result.codebook_gini[i]
        top1 = result.codebook_top1_concentration[i]
        lines.append(
            f"| {i} | {util:.4f} | {active:,} | {perplexity:.2f} | "
            f"{gini:.4f} | {top1:.4f} |"
        )
    lines.append("")

    lines.append("### Collision Bucket Quality\n")
    lines.append(f"- Collision buckets: {result.collision_bucket_count:,}")
    if result.collision_category_entropy_mean is not None:
        lines.append(
            "- Mean category entropy: "
            f"{result.collision_category_entropy_mean:.4f}"
        )
    if result.collision_geo_radius_mean_km is not None:
        lines.append(
            f"- Mean geo radius: {result.collision_geo_radius_mean_km:.4f} km"
        )
    if result.collision_geo_radius_p95_km is not None:
        lines.append(
            f"- P95 geo radius: {result.collision_geo_radius_p95_km:.4f} km"
        )
    lines.append("")

    if result.prefix_diversity is not None:
        lines.append("### Prefix Diversity\n")
        for depth, count in sorted(result.prefix_diversity.items()):
            lines.append(f"- Depth {depth}: {count:,} distinct")
        lines.append("")

    if result.prefix_category_purity is not None:
        lines.append("### Category Structure by SID Prefix\n")
        lines.append(
            "Prefix purity is POI-weighted; macro purity gives each prefix "
            "cluster equal weight. Entropy is size-weighted Shannon entropy "
            "in nats. Category-neighbour recall is the fraction of each "
            "POI's same-category peers retrieved by its prefix, excluding "
            "the POI itself.\n"
        )
        lines.append(
            "| Depth | Purity | Macro purity | Weighted entropy | AMI | NMI | "
            "Category-neighbour recall | Purity 95% CI | Candidate size | "
            "Geo radius (km) | Collision rate |"
        )
        lines.append(
            "|-------|--------|--------------|------------------|-----|-----|"
            "---------------------------|----------------|----------------|"
            "-----------------|---------------|"
        )
        depths = sorted(result.prefix_category_purity)
        for depth in depths:
            lines.append(
                f"| {depth} | "
                f"{result.prefix_category_purity[depth]:.4f} | "
                f"{_format_metric(result.prefix_category_macro_purity, depth)} | "
                f"{_format_metric(result.prefix_category_entropy_weighted, depth)} | "
                f"{_format_metric(result.prefix_category_ami, depth)} | "
                f"{_format_metric(result.prefix_category_nmi, depth)} | "
                f"{_format_metric(result.prefix_category_neighbor_recall, depth)} | "
                f"{_format_ci(result.prefix_category_purity_ci, depth)} | "
                f"{_format_metric(result.prefix_candidate_set_size_mean, depth)} | "
                f"{_format_metric(result.prefix_geo_radius_mean_km, depth)} | "
                f"{_format_metric(result.prefix_collision_rate, depth)} |"
            )
        lines.append("")

    if result.candidate_count_histogram:
        lines.append("### Ambiguity Distribution\n")
        lines.append("| Candidates | Count |")
        lines.append("|------------|-------|")
        for n_cand, count in sorted(result.candidate_count_histogram.items()):
            lines.append(f"| {n_cand} | {count:,} |")
        lines.append("")

    if result.knn_recall_at_k is not None:
        lines.append("## CF Alignment\n")
        lines.append(
            f"- k-NN recall@{result.knn_k} (H<=1): "
            f"{result.knn_recall_at_k:.4f}"
        )
        if result.knn_recall_at_k_h2 is not None:
            lines.append(
                f"- k-NN recall@{result.knn_k} (H<=2): "
                f"{result.knn_recall_at_k_h2:.4f}"
            )
        if result.knn_recall_at_k_h3 is not None:
            lines.append(
                f"- k-NN recall@{result.knn_k} (H<=3): "
                f"{result.knn_recall_at_k_h3:.4f}"
            )
        if result.prefix_knn_recall_at_k is not None:
            lines.append(
                f"- k-NN recall@{result.knn_k} (prefix>=1): "
                f"{result.prefix_knn_recall_at_k:.4f}"
            )
        if result.prefix_knn_recall_at_k_by_depth is not None:
            lines.append("### CF Neighbour Recall by SID Prefix Depth\n")
            for depth, value in sorted(
                result.prefix_knn_recall_at_k_by_depth.items()
            ):
                lines.append(
                    f"- Depth {depth}: recall@{result.knn_k} = {value:.4f}"
                )
            lines.append("")
        if result.cf_transition_recall_at_k is not None:
            lines.append(
                f"- Collapsed-transition CF recall@{result.knn_k}: "
                f"{result.cf_transition_recall_at_k:.4f}"
            )
        if result.cosine_hamming_spearman is not None:
            lines.append(
                "- Cosine-Hamming Spearman: "
                f"{result.cosine_hamming_spearman:.4f}"
            )
        if result.weighted_hamming_spearman is not None:
            lines.append(
                "- Weighted-Hamming Spearman: "
                f"{result.weighted_hamming_spearman:.4f}"
            )
        if result.prefix_cf_coherence is not None:
            baseline_str = (
                f"{result.prefix_cf_coherence_baseline:.4f}"
                if result.prefix_cf_coherence_baseline is not None
                else "n/a"
            )
            lines.append(
                f"- Prefix CF coherence: "
                f"{result.prefix_cf_coherence:.4f} "
                f"(random baseline: {baseline_str})"
            )
        lines.append(f"- Sample size: {result.cf_sample_size:,}")
        if result.cf_pair_sample_size:
            lines.append(
                f"- Pairwise correlation pairs: {result.cf_pair_sample_size:,}"
            )
        lines.append("")

    if result.spatial_displacement_median_km is not None:
        lines.append("## Spatial Displacement\n")
        lines.append(
            f"- Median displacement: "
            f"{result.spatial_displacement_median_km:.2f} km"
        )
        lines.append("")

    if result.summed_space_metrics_by_depth is not None:
        lines.append("## Flow-Consumed Summed SID Geometry\n")
        lines.append(
            "At depth `d`, the flow input is `z_i(d) = sum_{l=1}^d "
            "E_l[c_{i,l}]`. Neighbours below use Euclidean distance in "
            "that unnormalised summed space; graph neighbours use cosine "
            "similarity in the configured graph/CF embedding. Temporal "
            "transitions collapse consecutive identical POIs first.\n"
        )
        lines.append(
            f"- Evaluated POIs: {result.summed_space_evaluated_pois:,}; "
            f"neighbourhood k: {result.summed_space_k}"
        )
        lines.append(
            "| Depth | Graph-neighbour recall | Transition-neighbour recall | "
            "Category purity | Temporal-profile cosine | "
            "Mean geographic distance (km) |"
        )
        lines.append(
            "|-------|------------------------|----------------------------|"
            "-----------------|-------------------------|-----------------------------|"
        )
        for depth in range(1, result.n_codebooks + 1):
            metrics = result.summed_space_metrics_by_depth.get(depth, {})
            lines.append(
                f"| {depth} | "
                f"{_format_metric(metrics, 'graph_neighbor_recall_at_k')} | "
                f"{_format_metric(metrics, 'temporal_transition_neighbor_recall_at_k')} | "
                f"{_format_metric(metrics, 'category_purity_at_k')} | "
                f"{_format_metric(metrics, 'temporal_profile_similarity_at_k')} | "
                f"{_format_metric(metrics, 'geographic_neighbor_distance_km')} |"
            )
        lines.append("")

        lines.append("### Prefix-Geometry Consistency\n")
        lines.append(
            "`D_d` is the mean full-depth summed-vector distance among POI "
            "pairs with longest common prefix exactly `d`. A strict additive "
            "hierarchy should satisfy `D_0 > D_1 > ... > D_(K-1)`.\n"
        )
        status = (
            "passes"
            if result.prefix_geometry_strictly_decreasing
            else "does not pass"
        )
        if result.prefix_geometry_strictly_decreasing is None:
            status = "not evaluated"
        lines.append(f"- Strict-decrease check: {status}")
        if result.prefix_geometry_monotonicity_violations:
            violations = ", ".join(
                f"D_{depth} <= D_{depth + 1}"
                for depth in result.prefix_geometry_monotonicity_violations
            )
            lines.append(f"- Violations: {violations}")
        lines.append("| Exact LCP | D_d | Sampled pairs | Eligible pairs |")
        lines.append("|-----------|-----|---------------|----------------|")
        for depth in range(result.n_codebooks):
            lines.append(
                f"| {depth} | "
                f"{_format_metric(result.prefix_geometry_lcp_distance, depth)} | "
                f"{_format_count_metric(result.prefix_geometry_lcp_pair_count, depth)} | "
                f"{_format_count_metric(result.prefix_geometry_lcp_total_pairs, depth)} |"
            )
        lines.append("")

        lines.append("### Per-Level Residual Scale\n")
        lines.append(
            "The variance columns are computed over the selected residual "
            "vectors across the evaluated POI catalog; total variance is "
            "the covariance trace.\n"
        )
        lines.append(
            "| Level | Mean selected-vector norm | Mean coordinate variance | "
            "Total variance |"
        )
        lines.append(
            "|-------|---------------------------|--------------------------|----------------|"
        )
        for depth in range(1, result.n_codebooks + 1):
            lines.append(
                f"| {depth} | "
                f"{_format_metric(result.per_level_component_mean_norm, depth)} | "
                f"{_format_metric(result.per_level_component_coordinate_variance, depth)} | "
                f"{_format_metric(result.per_level_component_total_variance, depth)} |"
            )
        lines.append("")

    if result.poi_sid_only_top1_recovery_rate is not None:
        lines.append("## POI Round-Trip Recovery\n")
        lines.append(
            "Recovery is reported separately for a deterministic smallest-ID "
            "SID-only "
            "tie-break, candidate-set coverage, and optional embedding help.\n"
        )
        lines.append(
            "- Exact top-1 SID-only recovery: "
            f"{result.poi_sid_only_top1_recovery_rate:.4f}"
        )
        lines.append(
            "- Candidate-set recovery: "
            f"{result.poi_candidate_set_recovery_rate:.4f}"
        )
        lines.append(
            "- SID-only top-1 recovery excluding collisions: "
            f"{result.poi_sid_only_top1_recovery_noncollision_rate:.4f}"
        )
        if result.poi_sid_only_topk_recovery_rates is not None:
            for k, value in sorted(
                result.poi_sid_only_topk_recovery_rates.items()
            ):
                lines.append(
                    f"- SID-only candidate-list recovery@{k}: {value:.4f}"
                )
        if result.poi_embedding_assisted_recovery_rate is None:
            lines.append("- Embedding-assisted top-1 recovery: not evaluated")
        else:
            lines.append(
                "- Embedding-assisted top-1 recovery: "
                f"{result.poi_embedding_assisted_recovery_rate:.4f} "
                f"(scored {result.poi_embedding_assisted_scored:,} POIs)"
            )
        lines.append("")

    if result.baseline_metrics:
        lines.append("## Controlled Baselines\n")
        lines.append(
            "All rows use the same POI catalog, stays, CF artifact, prefix "
            "depths, and bootstrap settings as the reported tokenizer. "
            "Raw IDs and random balanced codes are hierarchy controls; "
            "category and geohash cells are strong simple urban-mobility "
            "controls.\n"
        )
        lines.append(
            "| Baseline | Purity d1 | Purity d2 | Purity d3 | "
            "Transition entropy d1 | Temporal coherence d1 | "
            "CF neighbour recall d1 | Next-prefix accuracy d1 |"
        )
        lines.append(
            "|----------|-----------|-----------|-----------|----------------------|"
            "------------------------|--------------------------|--------------------------|"
        )
        for name, metrics in sorted(result.baseline_metrics.items()):
            lines.append(
                f"| {name} | "
                f"{_format_metric(metrics.get('prefix_category_purity'), 1)} | "
                f"{_format_metric(metrics.get('prefix_category_purity'), 2)} | "
                f"{_format_metric(metrics.get('prefix_category_purity'), 3)} | "
                f"{_format_metric(metrics.get('transition_entropy_by_prefix'), 1)} | "
                f"{_format_metric(metrics.get('prefix_temporal_coherence_by_depth'), 1)} | "
                f"{_format_metric(metrics.get('prefix_knn_recall_at_k_by_depth'), 1)} | "
                f"{_format_metric(metrics.get('next_prefix_prediction_accuracy'), 1)} |"
            )
        lines.append("")

    if result.bootstrap_confidence_intervals:
        lines.append("## Bootstrap Uncertainty\n")
        lines.append(
            "Reported intervals are percentile bootstrap 95% CIs. POI "
            "purity intervals condition on the observed prefix assignments; "
            "user, trajectory, and time-period intervals resample those "
            "clusters, so repeated stay rows are not treated as independent.\n"
        )
        lines.append(
            "| Unit | Metric | Depth 1 CI | Depth 2 CI | Depth 3 CI |"
        )
        lines.append(
            "|------|--------|------------|------------|------------|"
        )
        for unit in ("poi", "users", "trajectories", "time_periods"):
            payload = result.bootstrap_confidence_intervals.get(unit)
            if unit == "poi":
                metric_name = "prefix category purity"
                values = payload
            else:
                metric_name = "prefix sharing rate"
                values = payload if payload else None
            lines.append(
                f"| {unit} | {metric_name} | "
                f"{_format_ci(values, 1)} | {_format_ci(values, 2)} | "
                f"{_format_ci(values, 3)} |"
            )
        effect_values = result.bootstrap_confidence_intervals.get(
            "temporal_coherence_difference"
        )
        if effect_values:
            lines.append("")
            lines.append(
                "Temporal coherence effect (within-prefix JSD minus random "
                "profile baseline) CIs: "
                + ", ".join(
                    f"d{depth} {_format_ci(effect_values, depth)}"
                    for depth in sorted(effect_values)
                )
                + "."
            )
        lines.append("")

    has_traj = (
        result.temporal_distinctiveness is not None
        or result.transition_entropy is not None
        or result.prefix_sharing_rate is not None
        or result.prefix_temporal_coherence is not None
        or result.transition_entropy_by_prefix is not None
        or result.next_prefix_prediction_accuracy is not None
        or result.anchor_repetition_rate is not None
    )
    if has_traj:
        lines.append("## Trajectory-Level Behavioral Metrics\n")
        lines.append(
            "These metrics evaluate whether SID codes capture meaningful "
            "behavioral structure by analyzing real user trajectories "
            "(movement sequences across POIs). Unlike POI-level metrics "
            "above, these require stay-level data and measure how SIDs "
            "behave in the context of actual mobility patterns.\n"
        )
        if result.temporal_distinctiveness is not None:
            lines.append("### Temporal Distinctiveness\n")
            lines.append(
                "Mean Jensen-Shannon divergence between each SID's "
                "hourly visit distribution and the global hourly "
                "distribution. Higher values mean SIDs encode "
                "time-of-day semantics (e.g., nightlife vs. morning "
                "commute POIs get distinct codes). A value near 0 means "
                "SIDs are temporally indistinguishable from random.\n"
            )
            lines.append(f"- **{result.temporal_distinctiveness:.4f}**\n")
        if result.transition_entropy is not None:
            lines.append("### Transition Entropy\n")
            lines.append(
                "Mean normalized entropy of per-SID outgoing transition "
                "distributions. Measures whether movement from a given "
                "SID goes to a predictable set of next SIDs (low entropy) "
                "or scatters uniformly (high entropy, ~1.0). Lower values "
                "indicate the code space captures structured mobility "
                "patterns.\n"
            )
            lines.append(f"- **{result.transition_entropy:.4f}**\n")
        if result.transition_entropy_by_prefix is not None:
            lines.append("### Transition Entropy by SID Prefix Depth\n")
            lines.append(
                "Transitions are computed after collapsing consecutive "
                "identical POIs; distinct POIs sharing a SID remain movement.\n"
            )
            for depth, value in sorted(
                result.transition_entropy_by_prefix.items()
            ):
                lines.append(f"- Depth {depth}: {value:.4f}")
            lines.append("")
        if result.prefix_sharing_rate is not None:
            lines.append("### Prefix Sharing Rate\n")
            lines.append(
                "Fraction of consecutive visits within the same "
                "trajectory that share a SID prefix at each codebook "
                "depth. Higher rates mean the hierarchical prefix "
                "structure captures spatial or behavioral locality "
                "within trips (e.g., nearby POIs visited in sequence "
                "share a coarse code prefix).\n"
            )
            for depth, rate in sorted(result.prefix_sharing_rate.items()):
                lines.append(f"- Depth {depth}: {rate:.4f}")
            lines.append("")
        if result.prefix_temporal_coherence_by_depth is not None:
            lines.append("### Prefix Temporal Coherence by Depth\n")
            lines.append(
                "Mean pairwise hourly-profile JSD among distinct POIs sharing "
                "a prefix, after same-POI collapse. Lower is more coherent.\n"
            )
            lines.append(
                "| Depth | Coherence | Random baseline | Difference | 95% CI |"
            )
            lines.append(
                "|-------|-----------|-----------------|------------|--------|"
            )
            for depth in range(1, result.n_codebooks + 1):
                value = result.prefix_temporal_coherence_by_depth.get(depth)
                baseline = _format_metric(
                    result.prefix_temporal_coherence_baseline_by_depth,
                    depth,
                )
                difference = (
                    f"{value - result.prefix_temporal_coherence_baseline_by_depth[depth]:.4f}"
                    if value is not None
                    and result.prefix_temporal_coherence_baseline_by_depth
                    and depth
                    in result.prefix_temporal_coherence_baseline_by_depth
                    else "n/a"
                )
                lines.append(
                    f"| {depth} | {_format_metric(result.prefix_temporal_coherence_by_depth, depth)} | "
                    f"{baseline} | {difference} | "
                    f"{_format_ci(result.prefix_temporal_coherence_effect_ci_by_depth, depth)} |"
                )
            lines.append("")
        elif result.prefix_temporal_coherence is not None:
            baseline_str = (
                f"{result.prefix_temporal_coherence_baseline:.4f}"
                if result.prefix_temporal_coherence_baseline is not None
                else "n/a"
            )
            lines.append("### Prefix Temporal Coherence\n")
            lines.append(
                "Mean pairwise hourly-distribution JSD among SIDs that "
                "share the same first-codebook prefix. Lower values mean "
                "the coarsest codebook groups POIs with similar temporal "
                "usage (the prefix is temporally meaningful). Compare "
                "against the random baseline: if coherence is well below "
                "baseline, the prefix captures real temporal structure "
                "rather than noise.\n"
            )
            lines.append(
                f"- **{result.prefix_temporal_coherence:.4f}** "
                f"(random baseline: {baseline_str})\n"
            )
        if result.next_prefix_prediction_accuracy is not None:
            lines.append("### Next-Prefix Prediction and Next-POI MI\n")
            lines.append(
                "Prediction accuracy is the empirical modal next-prefix "
                "accuracy among eligible source prefixes. MI is in nats.\n"
            )
            lines.append(
                "| Depth | Modal next-prefix accuracy | MI with next POI |"
            )
            lines.append(
                "|-------|----------------------------|------------------|"
            )
            for depth, value in sorted(
                result.next_prefix_prediction_accuracy.items()
            ):
                lines.append(
                    f"| {depth} | {value:.4f} | "
                    f"{_format_metric(result.next_poi_mutual_information, depth)} |"
                )
            lines.append("")
        if result.anchor_repetition_rate is not None:
            lines.append(
                "- Anchor repetition `Pr(P[t+1] = P[t])`: "
                f"{result.anchor_repetition_rate:.4f}"
            )
            if result.raw_event_count:
                lines.append(
                    f"- POI events after collapse: {result.collapsed_event_count:,} "
                    f"/ {result.raw_event_count:,}"
                )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["write_semantic_id_eval_report"]
