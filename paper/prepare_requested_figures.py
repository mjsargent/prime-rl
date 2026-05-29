from __future__ import annotations

import json
from pathlib import Path

PLOT = '''from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figure_utils import build_figure


if __name__ == "__main__":
    build_figure(Path(__file__).parent)
'''


FIGURES = {
    "stage1_chart_invariance": {
        "figure_type": "chart_invariance",
        "title": "Stage 1 Chart Invariance",
        "source_run_ids": [
            "stage1_v2_original_graph_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_original_graph_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage1_discretization_invariance": {
        "figure_type": "discretization_invariance",
        "title": "Stage 1 Discretization Invariance",
        "source_run_ids": [
            "stage1_v2_original_graph_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_original_graph_Qwen_Qwen3_8B_primeintellect_math500",
            "stage1_v2_graph_free_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_graph_free_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage1_sample_size_convergence": {
        "figure_type": "sample_size_convergence",
        "title": "Stage 1 Sample-Size Convergence",
        "source_run_ids": [
            "stage1_v2_original_graph_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_original_graph_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage1_layer_pair_heatmap": {
        "figure_type": "layer_pair_heatmap",
        "title": "Layer-Pair Effective Rank",
        "source_run_ids": [
            "stage1_v2_original_graph_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_original_graph_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage1_effective_rank_distribution": {
        "figure_type": "effective_rank_distribution",
        "title": "Effective Rank Distribution",
        "source_run_ids": [
            "stage1_v2_original_graph_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_original_graph_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage1_cumulative_gain_curve": {
        "figure_type": "cumulative_gain_curve",
        "title": "Cumulative Controllable Gain",
        "source_run_ids": [
            "stage1_v2_original_graph_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_original_graph_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage1_cross_formulation_comparison": {
        "figure_type": "cross_formulation_comparison",
        "title": "Cross-Formulation Construction Evidence",
        "source_run_ids": [
            "stage1_v2_graph_free_Qwen_Qwen3_8B_prime_swe_grep",
            "stage1_v2_graph_free_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage2_null_intervention_distribution": {
        "figure_type": "stage2_null",
        "title": "Null-Intervention Encoder Shift",
        "source_run_ids": [
            "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b",
            "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b",
        ],
    },
    "stage2_random_baseline_calibration": {
        "figure_type": "stage2_random",
        "title": "Random-Baseline Calibration",
        "source_run_ids": [
            "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b",
            "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b",
        ],
    },
    "stage2_baseline_identifiability_profile": {
        "figure_type": "stage2_baseline_profile",
        "title": "Chance-Corrected Baseline Identifiability",
        "source_run_ids": [
            "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b",
            "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b",
        ],
    },
    "stage2_baseline_calibration": {
        "figure_type": "stage2_random",
        "title": "Stage 2 Baseline Calibration",
        "source_run_ids": [
            "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b",
            "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b",
        ],
    },
    "stage3_per_coordinate_geometry": {
        "figure_type": "stage3_scatter",
        "title": "Per-Coordinate Reachability and Gain",
        "source_run_ids": [
            "stage3_v2_Qwen_Qwen3_8B_prime_swe_grep",
            "stage3_v2_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage3_linearization_budget": {
        "figure_type": "stage3_eta",
        "title": "Linearization Budget by Coordinate",
        "source_run_ids": [
            "stage3_v2_Qwen_Qwen3_8B_prime_swe_grep",
            "stage3_v2_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage3_reachability_distribution": {
        "figure_type": "stage3_reachability",
        "title": "Reachability Distribution by Coordinate",
        "source_run_ids": [
            "stage3_v2_Qwen_Qwen3_8B_prime_swe_grep",
            "stage3_v2_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage3_controllability_spectrum": {
        "figure_type": "stage3_spectrum",
        "title": "Controllability Spectrum",
        "source_run_ids": [
            "stage3_v2_Qwen_Qwen3_8B_prime_swe_grep",
            "stage3_v2_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage3_cross_formulation_per_coordinate": {
        "figure_type": "stage3_cross_formulation",
        "title": "Cross-Formulation Per-Coordinate Gain",
        "source_run_ids": [
            "stage3_v2_Qwen_Qwen3_8B_prime_swe_grep",
            "stage3_v2_Qwen_Qwen3_8B_primeintellect_math500",
        ],
    },
    "stage4_trust_region_violation_rate": {
        "figure_type": "stage4_trust_region",
        "title": "Trust-Region Violation Rate by Edit Norm",
        "source_run_ids": [
            "stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_pilot_rerun3",
            "stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_pilot_rerun3",
        ],
    },
    "stage4_realized_vs_predicted_displacement": {
        "figure_type": "stage4_displacement",
        "title": "Realized vs Predicted Displacement",
        "source_run_ids": [
            "stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_pilot_rerun3",
            "stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_pilot_rerun3",
        ],
    },
    "stage4_trajectory_length_by_controller": {
        "figure_type": "stage4_length",
        "title": "Trajectory Length by Controller",
        "source_run_ids": [
            "stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_pilot_rerun3",
            "stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_pilot_rerun3",
        ],
    },
    "stage4_coherence_rate_by_edit_norm": {
        "figure_type": "stage4_coherence",
        "title": "Coherence Rate by Edit Norm",
        "source_run_ids": [
            "stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_pilot_rerun3",
            "stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_pilot_rerun3",
        ],
    },
    "stage5_identifiability_bars": {
        "figure_type": "stage5_identifiability_bars",
        "title": "Stage 5 Identifiability vs Baselines",
        "source_run_ids": [
            "stage5_v2_Qwen_Qwen3_8B_tier_a_stage5",
            "stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b",
            "stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b",
        ],
    },
    "stage5_per_coordinate_identifiability": {
        "figure_type": "stage5_per_coordinate_identifiability",
        "title": "Per-Coordinate Identifiability",
        "source_run_ids": [
            "stage5_v2_Qwen_Qwen3_8B_tier_a_stage5",
        ],
    },
    "stage5_cross_formulation_identifiability": {
        "figure_type": "stage5_cross_formulation_identifiability",
        "title": "Cross-Formulation Identifiability",
        "source_run_ids": [
            "stage5_v2_Qwen_Qwen3_8B_tier_a_stage5",
        ],
    },
    "stage5_confusion_matrix_headline": {
        "figure_type": "stage5_confusion_matrix_headline",
        "title": "Parametric-B Confusion Matrices",
        "source_run_ids": [
            "stage5_v2_Qwen_Qwen3_8B_tier_a_stage5",
        ],
    },
    "stage5_quality_conditioning_sensitivity": {
        "figure_type": "stage5_quality_conditioning_sensitivity",
        "title": "Quality-Conditioning Sensitivity",
        "source_run_ids": [
            "stage5_v2_Qwen_Qwen3_8B_tier_a_stage5",
        ],
    },
}


def main() -> None:
    root = Path(__file__).resolve().parent / "figures"
    for name, manifest in FIGURES.items():
        fig_dir = root / name
        fig_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "envs": ["swe", "math500"],
            **manifest,
        }
        (fig_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (fig_dir / "plot.py").write_text(PLOT)


if __name__ == "__main__":
    main()
