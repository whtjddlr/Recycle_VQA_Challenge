#!/usr/bin/env python3
"""
Colab-friendly three-pass multistage runner.

Typical Colab usage:

    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/2026-ssafy-ai-15-2
    !python colab_three_pass_multistage.py \
        --project-root /content/drive/MyDrive/2026-ssafy-ai-15-2 \
        --source test \
        --html-subset /content/drive/MyDrive/final_subset.csv

For a plain full-test run, omit --html-subset.
"""

from __future__ import annotations

import argparse
import importlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

import multistage_vqa

# During notebook-style iteration we want the latest local edits immediately.
importlib.reload(multistage_vqa)

from multistage_vqa import (
    CHOICE_LABELS,
    MultiStageConfig,
    MultiStageVQAPipeline,
    build_brain_choice_prior,
    build_choices,
    build_detection_choice_prior,
    build_detection_count_prior,
    classify_question_type,
    clear_runtime_memory,
    load_brain_plan_cache,
    load_context_cache,
)


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sanitize_tag(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "subset"


def build_detector_policy_configs(
    base_grounding_model_id: Optional[str],
    base_sam_model_id: Optional[str],
) -> Dict[str, Dict]:
    # Each qtype gets its own detector policy so the pipeline can be aggressive
    # for material/recycle, conservative for count, and disabled for color/state.
    return {
        "material": {
            "name": "waste_specialist",
            "description": "재질 문제라 waste prompt와 choice-aware grounding을 적극 사용합니다.",
            "overrides": {
                "grounding_model_id": base_grounding_model_id,
                "sam_model_id": base_sam_model_id,
                "grounding_prompt_strategy": "robust_waste",
                "grounding_use_choice_prompts": True,
                "grounding_choice_prompt_limit": 4,
                "grounding_query_limit": 18,
                "grounding_threshold": 0.22,
                "grounding_text_threshold": 0.18,
                "grounding_iou_threshold": 0.40,
                "grounding_max_box_area_ratio": 0.72,
                "max_boxes": 5,
                "detection_choice_prior_weight": 0.18,
                "detection_count_prior_weight": 0.0,
            },
        },
        "recycle": {
            "name": "waste_specialist",
            "description": "분리배출/분류 문제라 waste detector 신호를 강하게 봅니다.",
            "overrides": {
                "grounding_model_id": base_grounding_model_id,
                "sam_model_id": base_sam_model_id,
                "grounding_prompt_strategy": "robust_waste",
                "grounding_use_choice_prompts": True,
                "grounding_choice_prompt_limit": 4,
                "grounding_query_limit": 18,
                "grounding_threshold": 0.22,
                "grounding_text_threshold": 0.18,
                "grounding_iou_threshold": 0.40,
                "grounding_max_box_area_ratio": 0.72,
                "max_boxes": 5,
                "detection_choice_prior_weight": 0.20,
                "detection_count_prior_weight": 0.0,
            },
        },
        "type": {
            "name": "mixed_detector",
            "description": "종류 문제라 detector는 쓰되, 최종 판단은 student VLM 비중을 더 남겨둡니다.",
            "overrides": {
                "grounding_model_id": base_grounding_model_id,
                "sam_model_id": base_sam_model_id,
                "grounding_prompt_strategy": "robust_waste",
                "grounding_use_choice_prompts": True,
                "grounding_choice_prompt_limit": 3,
                "grounding_query_limit": 12,
                "grounding_threshold": 0.24,
                "grounding_text_threshold": 0.18,
                "max_boxes": 4,
                "detection_choice_prior_weight": 0.10,
                "detection_count_prior_weight": 0.0,
            },
        },
        "dominant": {
            "name": "mixed_detector",
            "description": "가장 많은 종류 비교라 detector 후보는 넓게 보되 prior는 약하게만 씁니다.",
            "overrides": {
                "grounding_model_id": base_grounding_model_id,
                "sam_model_id": base_sam_model_id,
                "grounding_prompt_strategy": "robust_waste",
                "grounding_use_choice_prompts": True,
                "grounding_choice_prompt_limit": 3,
                "grounding_query_limit": 14,
                "grounding_threshold": 0.24,
                "grounding_text_threshold": 0.18,
                "max_boxes": 6,
                "detection_choice_prior_weight": 0.08,
                "detection_count_prior_weight": 0.0,
            },
        },
        "count": {
            "name": "count_conservative",
            "description": "개수 문제라 choice prompt는 끄고, 더 많은 박스를 보존하도록 보수적으로 탐지합니다.",
            "overrides": {
                "grounding_model_id": base_grounding_model_id,
                "sam_model_id": base_sam_model_id,
                "grounding_prompt_strategy": "focus_only",
                "grounding_use_choice_prompts": False,
                "grounding_query_limit": 10,
                "grounding_threshold_count": 0.32,
                "grounding_iou_threshold_count": 0.24,
                "grounding_max_box_area_ratio_count": 0.35,
                "detection_quality_threshold_for_prior": 0.50,
                "max_boxes": 20,
                "panel_tile_size": 220,
                "detection_choice_prior_weight": 0.0,
                "detection_count_prior_weight": 0.28,
            },
            "count_detector_blend_weight": 0.55,
        },
        "location": {
            "name": "focus_only_localizer",
            "description": "위치 문제라 choice prompt는 끄고, 대상 위치 추정용 detector만 가볍게 사용합니다.",
            "overrides": {
                "grounding_model_id": base_grounding_model_id,
                "sam_model_id": base_sam_model_id,
                "grounding_prompt_strategy": "focus_only",
                "grounding_use_choice_prompts": False,
                "grounding_query_limit": 8,
                "grounding_threshold": 0.28,
                "grounding_text_threshold": 0.20,
                "max_boxes": 3,
                "detection_choice_prior_weight": 0.0,
                "detection_count_prior_weight": 0.0,
            },
        },
        "color": {
            "name": "vlm_first",
            "description": "색상 문제라 waste detector 도움보다 student VLM 판단을 우선합니다.",
            "overrides": {
                "grounding_model_id": None,
                "sam_model_id": None,
                "grounding_use_choice_prompts": False,
                "detection_choice_prior_weight": 0.0,
                "detection_count_prior_weight": 0.0,
            },
        },
        "state": {
            "name": "vlm_first",
            "description": "상태/라벨/문구 문제라 detector를 끄고 원본 이미지 해석에 맡깁니다.",
            "overrides": {
                "grounding_model_id": None,
                "sam_model_id": None,
                "grounding_use_choice_prompts": False,
                "detection_choice_prior_weight": 0.0,
                "detection_count_prior_weight": 0.0,
            },
        },
        "other": {
            "name": "vlm_first",
            "description": "도메인 밖 질문은 detector를 끄고 brain + student 위주로 풉니다.",
            "overrides": {
                "grounding_model_id": None,
                "sam_model_id": None,
                "grounding_use_choice_prompts": False,
                "detection_choice_prior_weight": 0.0,
                "detection_count_prior_weight": 0.0,
            },
        },
    }


def select_detector_policy(
    question: str,
    choice_map: Dict[str, str],
    detector_policy_configs: Dict[str, Dict],
    planned_question_type: Optional[str] = None,
) -> Tuple[str, Dict]:
    question_type = str(planned_question_type or classify_question_type(str(question), choice_map)).strip().lower()
    if not question_type:
        question_type = classify_question_type(str(question), choice_map)
    return question_type, detector_policy_configs.get(question_type, detector_policy_configs["other"])


def apply_policy_to_config(config: MultiStageConfig, policy: Dict) -> Dict[str, object]:
    overrides = dict(policy["overrides"])
    original = {key: getattr(config, key) for key in overrides}
    for key, value in overrides.items():
        setattr(config, key, value)
    return original


def restore_policy_to_config(config: MultiStageConfig, original: Dict[str, object]) -> None:
    for key, value in original.items():
        setattr(config, key, value)


def normalize_probs(choice_probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(choice_probs.values()) or 1.0
    return {label: value / total for label, value in choice_probs.items()}


def build_detector_biased_probs(
    raw_probs: Dict[str, float],
    count_prior: Optional[Dict[str, float]],
    count_prior_weight: float,
    detector_policy: Dict,
    question_type: str,
) -> Tuple[Dict[str, float], str, float, bool, float]:
    # Count is the only place where we post-blend the student distribution with
    # a detector-derived prior. All other qtypes keep the raw student output.
    adjusted = dict(raw_probs)
    policy_blend_weight = float(detector_policy.get("count_detector_blend_weight", 0.0))
    blend_weight = min(policy_blend_weight, max(0.0, float(count_prior_weight or 0.0)))
    applied = False
    if question_type == "count" and count_prior and blend_weight > 0:
        adjusted = normalize_probs(
            {
                label: raw_probs[label] * (1.0 - blend_weight) + count_prior[label] * blend_weight
                for label in CHOICE_LABELS
            }
        )
        applied = True
    answer = max(adjusted.items(), key=lambda item: item[1])[0]
    confidence = adjusted[answer]
    return adjusted, answer, confidence, applied, blend_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multistage pipeline in three passes on Colab.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root containing CSVs, images, and multistage_vqa.py")
    parser.add_argument("--source", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--html-subset", type=Path, default=None, help="Optional HTML file whose test_*.jpg entries define a subset")
    parser.add_argument("--subset-ids-path", type=Path, default=None, help="Optional text file to save/read extracted subset ids")
    parser.add_argument("--subset-tag", type=str, default=None, help="Optional tag added to output filenames")
    parser.add_argument("--brain-model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--grounding-model-id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model-id", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--enable-sam", action="store_true")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional LoRA adapter path for the Qwen student")
    parser.add_argument("--enable-internvl", action="store_true", help="Enable the InternVL student backend")
    parser.add_argument(
        "--internvl-adapter-path",
        type=Path,
        default=None,
        help="Optional LoRA adapter path for the InternVL student",
    )
    parser.add_argument("--force-refresh-brain", action="store_true")
    parser.add_argument("--force-refresh-context", action="store_true")
    parser.add_argument("--force-rerun-student", action="store_true")
    parser.add_argument("--batch-save-every", type=int, default=50)
    return parser.parse_args()


def read_sample_df(project_root: Path, source: str) -> pd.DataFrame:
    source_to_file = {
        "train": project_root / "train.csv",
        "valid": project_root / "valid_part.csv",
        "test": project_root / "test.csv",
    }
    sample_df = pd.read_csv(source_to_file[source])
    if "id" not in sample_df.columns:
        sample_df["id"] = range(len(sample_df))
    sample_df["id"] = sample_df["id"].astype(str)
    return sample_df


def apply_html_subset(
    sample_df: pd.DataFrame,
    html_subset_path: Path,
    subset_ids_path: Path,
) -> pd.DataFrame:
    if subset_ids_path.exists():
        subset_ids = [line.strip() for line in subset_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        html_text = html_subset_path.read_text(encoding="utf-8", errors="ignore")
        subset_ids = sorted(set(re.findall(r"test_\d+\.jpg", html_text)))
        subset_ids_path.parent.mkdir(parents=True, exist_ok=True)
        subset_ids_path.write_text("\n".join(subset_ids) + "\n", encoding="utf-8")

    subset_set = set(subset_ids)
    return sample_df[
        sample_df["id"].isin(subset_set)
        | sample_df["path"].astype(str).map(lambda path: Path(path).name).isin(subset_set)
    ].reset_index(drop=True)


def order_prediction_df(df: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    ordered_ids = {str(row_id): idx for idx, row_id in enumerate(sample_ids)}
    ordered = df.copy()
    ordered["id"] = ordered["id"].astype(str)
    ordered["__order"] = ordered["id"].map(ordered_ids)
    ordered = ordered.sort_values("__order", kind="stable").drop(columns="__order")
    return ordered


def save_prediction_outputs(
    rows: Sequence[Dict[str, object]],
    output_full_csv: Path,
    output_answer_csv: Path,
    sample_ids: Sequence[str],
) -> pd.DataFrame:
    prediction_df = order_prediction_df(pd.DataFrame(rows), sample_ids)
    prediction_df.to_csv(output_full_csv, index=False, encoding="utf-8-sig")
    prediction_df[["id", "answer"]].to_csv(output_answer_csv, index=False, encoding="utf-8-sig")
    return prediction_df


def bool_override(policy: Dict, key: str, default: object) -> bool:
    return bool(policy["overrides"].get(key, default))


def build_pipeline_config(args: argparse.Namespace, project_root: Path, device: str) -> MultiStageConfig:
    use_4bit = device == "cuda"
    base_cfg = MultiStageConfig()
    cfg = MultiStageConfig(
        brain_model_id=args.brain_model_id,
        local_brain_fallback_model_id="Qwen/Qwen3-VL-4B-Instruct",
        prefer_local_brain_fallback_on_non_cuda=True,
        grounding_model_id=args.grounding_model_id,
        sam_model_id=args.sam_model_id if args.enable_sam else None,
        brain_quant_4bit=use_4bit,
        brain_cache_dir=str(project_root / "artifacts" / "brain_cache"),
        panel_dir=str(project_root / "artifacts" / "focus_panels"),
        cache_dir=str(project_root / "artifacts" / "context_cache"),
        qwen_student=base_cfg.qwen_student,
        internvl_student=base_cfg.internvl_student,
    )
    cfg.qwen_student.quant_4bit = use_4bit
    cfg.internvl_student.quant_4bit = use_4bit
    # Qwen and InternVL adapters are wired independently so either branch can
    # run as base-only or adapter-augmented.
    if args.adapter_path:
        cfg.qwen_student.adapter_path = str(args.adapter_path)
    if args.enable_internvl or args.internvl_adapter_path:
        cfg.internvl_student.enabled = True
    if args.internvl_adapter_path:
        cfg.internvl_student.adapter_path = str(args.internvl_adapter_path)
    return cfg


def resolve_sample_paths(sample_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """Rewrite relative image paths so the clean folder can run from anywhere."""
    resolved = sample_df.copy()
    if "path" not in resolved.columns:
        return resolved

    def _resolve(path_value: object) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return raw
        path = Path(raw)
        if path.is_absolute():
            return str(path)
        return str((project_root / path).resolve())

    resolved["path"] = resolved["path"].map(_resolve)
    return resolved


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    device = get_default_device()
    print("project_root:", project_root)
    print("device:", device)

    sample_df = read_sample_df(project_root, args.source)

    subset_tag = args.subset_tag
    if args.html_subset:
        html_subset_path = args.html_subset.resolve()
        if subset_tag is None:
            subset_tag = sanitize_tag(html_subset_path.stem)
        subset_ids_path = args.subset_ids_path or (project_root / "artifacts" / f"{subset_tag}_ids.txt")
        sample_df = apply_html_subset(sample_df, html_subset_path, subset_ids_path.resolve())
        print("html_subset:", html_subset_path)
        print("subset_ids_path:", subset_ids_path)
    else:
        subset_ids_path = args.subset_ids_path

    sample_df = resolve_sample_paths(sample_df, project_root)

    print("source:", args.source)
    print("num_samples:", len(sample_df))
    if subset_tag:
        print("subset_tag:", subset_tag)

    input_stem = {
        "train": "train",
        "valid": "valid_part",
        "test": "test",
    }.get(args.source, "predictions")
    if subset_tag:
        input_stem = f"{input_stem}_{subset_tag}"

    output_brain_csv = project_root / f"{input_stem}_brain_plan.csv"
    output_context_csv = project_root / f"{input_stem}_context_summary.csv"
    output_detections_csv = project_root / f"{input_stem}_detections.csv"
    output_crops_csv = project_root / f"{input_stem}_crops.csv"
    output_full_csv = project_root / f"{input_stem}_predictions_full.csv"
    output_answer_csv = project_root / f"{input_stem}_answer.csv"

    force_refresh_brain = args.force_refresh_brain
    force_refresh_context = args.force_refresh_context
    force_rerun_student = args.force_rerun_student
    if force_refresh_brain and not force_refresh_context:
        print("force_refresh_brain=True -> context도 다시 만듭니다.")
        force_refresh_context = True
    if force_refresh_context and not force_rerun_student:
        print("force_refresh_context=True -> student도 다시 추론합니다.")
        force_rerun_student = True

    cfg = build_pipeline_config(args, project_root, device)
    detector_policy_configs = build_detector_policy_configs(cfg.grounding_model_id, cfg.sam_model_id)
    sample_ids = sample_df["id"].astype(str).tolist()

    # Pass 1: brain only
    cached_brain_map = load_brain_plan_cache(cfg.brain_cache_dir)
    pending_brain_indices = [
        idx for idx, row_id in enumerate(sample_ids)
        if force_refresh_brain or row_id not in cached_brain_map
    ]
    if pending_brain_indices:
        print(f"building missing brain plans: {len(pending_brain_indices)} / {len(sample_df)}")
        brain_cfg = deepcopy(cfg)
        brain_cfg.grounding_model_id = None
        brain_cfg.sam_model_id = None
        brain_cfg.qwen_student.enabled = False
        brain_cfg.internvl_student.enabled = False
        brain_pipeline = MultiStageVQAPipeline(brain_cfg)
        for batch_index in tqdm(pending_brain_indices, desc=f"Brain pass ({args.source})"):
            row = sample_df.iloc[batch_index]
            choice_map, _ = build_choices(row, shuffle=False)
            brain_pipeline.plan_row(row, force_refresh=force_refresh_brain, choice_map=choice_map)
        brain_pipeline._release_context_models()
        del brain_pipeline
        clear_runtime_memory()
    else:
        print(f"brain cache already complete: {len(sample_df)} / {len(sample_df)}")

    brain_map = load_brain_plan_cache(cfg.brain_cache_dir)
    brain_rows = []
    for batch_index in range(len(sample_df)):
        row = sample_df.iloc[batch_index]
        row_id = str(row["id"])
        choice_map, _ = build_choices(row, shuffle=False)
        heuristic_question_type = classify_question_type(str(row["question"]), choice_map)
        plan = brain_map.get(row_id)
        planned_question_type = plan.question_type if plan is not None else heuristic_question_type
        policy_question_type, detector_policy = select_detector_policy(
            row["question"],
            choice_map,
            detector_policy_configs,
            planned_question_type=planned_question_type,
        )
        brain_rows.append(
            {
                "id": row_id,
                "path": row.get("path", ""),
                "question": row.get("question", ""),
                "heuristic_question_type": heuristic_question_type,
                "brain_question_type": planned_question_type,
                "policy_question_type": policy_question_type,
                "policy_name": detector_policy["name"],
                "grounding_enabled": bool_override(detector_policy, "grounding_model_id", cfg.grounding_model_id),
                "focus_objects": "|".join(plan.focus_objects) if plan else "",
                "answer_strategy": plan.answer_strategy if plan else "",
                "brain_confidence": round(float(plan.confidence), 6) if plan else 0.0,
            }
        )
    pd.DataFrame(brain_rows).to_csv(output_brain_csv, index=False, encoding="utf-8-sig")
    print("saved brain csv      :", output_brain_csv)

    # Pass 2: context only
    cached_context_map = load_context_cache(cfg.cache_dir)
    pending_context_indices = [
        idx for idx, row_id in enumerate(sample_ids)
        if force_refresh_context or row_id not in cached_context_map
    ]
    if pending_context_indices:
        print(f"building missing contexts: {len(pending_context_indices)} / {len(sample_df)}")
        context_cfg = deepcopy(cfg)
        context_cfg.brain_model_id = None
        context_cfg.qwen_student.enabled = False
        context_cfg.internvl_student.enabled = False
        context_pipeline = MultiStageVQAPipeline(context_cfg)
        for batch_index in tqdm(pending_context_indices, desc=f"Context pass ({args.source})"):
            row = sample_df.iloc[batch_index]
            row_id = str(row["id"])
            choice_map, _ = build_choices(row, shuffle=False)
            plan = brain_map.get(row_id)
            planned_question_type = plan.question_type if plan is not None else classify_question_type(str(row["question"]), choice_map)
            _, detector_policy = select_detector_policy(
                row["question"],
                choice_map,
                detector_policy_configs,
                planned_question_type=planned_question_type,
            )
            original_config = apply_policy_to_config(context_cfg, detector_policy)
            try:
                context_pipeline.prepare_context(
                    row,
                    save_panel=True,
                    force_refresh=force_refresh_context,
                    plan=plan,
                )
            finally:
                restore_policy_to_config(context_cfg, original_config)
        context_pipeline._release_context_models()
        del context_pipeline
        clear_runtime_memory()
    else:
        print(f"context cache already complete: {len(sample_df)} / {len(sample_df)}")

    context_map = load_context_cache(cfg.cache_dir)
    context_rows = []
    detection_rows = []
    crop_rows = []
    for batch_index in range(len(sample_df)):
        row = sample_df.iloc[batch_index]
        row_id = str(row["id"])
        context = context_map.get(row_id)
        if context is None:
            continue
        choice_map, _ = build_choices(row, shuffle=False)
        plan = brain_map.get(row_id)
        planned_question_type = plan.question_type if plan is not None else context.question_type
        policy_question_type, detector_policy = select_detector_policy(
            row["question"],
            choice_map,
            detector_policy_configs,
            planned_question_type=planned_question_type,
        )
        context_rows.append(
            {
                "id": row_id,
                "path": row.get("path", ""),
                "question": row.get("question", ""),
                "brain_question_type": planned_question_type,
                "policy_question_type": policy_question_type,
                "policy_name": detector_policy["name"],
                "focus_objects": "|".join(context.focus_objects),
                "answer_strategy": context.answer_strategy,
                "brain_confidence": round(float(context.brain_confidence), 6),
                "grounding_quality": round(float(context.grounding_quality), 6),
                "grounding_skip_reason": context.grounding_skip_reason,
                "num_detections": len(context.detections),
                "num_crops": len(context.crops),
                "focus_panel_path": context.focus_panel_path or "",
            }
        )
        for det_index, det in enumerate(context.detections, start=1):
            detection_rows.append(
                {
                    "id": row_id,
                    "det_index": det_index,
                    "phrase": det.phrase,
                    "score": round(float(det.score), 6),
                    "label": det.label,
                    "choice_label": det.choice_label,
                    "source": det.source,
                    "x1": det.box[0],
                    "y1": det.box[1],
                    "x2": det.box[2],
                    "y2": det.box[3],
                }
            )
        for crop_index, crop in enumerate(context.crops, start=1):
            crop_rows.append(
                {
                    "id": row_id,
                    "crop_index": crop_index,
                    "phrase": crop.phrase,
                    "score": round(float(crop.score), 6),
                    "mask_score": round(float(crop.mask_score), 6),
                    "box_x1": crop.box[0],
                    "box_y1": crop.box[1],
                    "box_x2": crop.box[2],
                    "box_y2": crop.box[3],
                    "crop_x1": crop.crop_box[0],
                    "crop_y1": crop.crop_box[1],
                    "crop_x2": crop.crop_box[2],
                    "crop_y2": crop.crop_box[3],
                }
            )
    pd.DataFrame(context_rows).to_csv(output_context_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(detection_rows).to_csv(output_detections_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(crop_rows).to_csv(output_crops_csv, index=False, encoding="utf-8-sig")
    print("saved context csv    :", output_context_csv)
    print("saved detections csv :", output_detections_csv)
    print("saved crops csv      :", output_crops_csv)

    # Pass 3: student only
    # Brain and context are now frozen caches. This pass only loads the student
    # backends, scores choices, and optionally applies the count detector prior.
    existing_rows = []
    completed_prediction_ids = set()
    if output_full_csv.exists() and not force_rerun_student:
        existing_prediction_df = pd.read_csv(output_full_csv)
        if not existing_prediction_df.empty:
            existing_prediction_df["id"] = existing_prediction_df["id"].astype(str)
            existing_rows = existing_prediction_df.to_dict("records")
            completed_prediction_ids = set(existing_prediction_df["id"].tolist())
        print("existing predictions found:", len(completed_prediction_ids))

    pending_student_indices = [
        idx for idx, row_id in enumerate(sample_ids)
        if force_rerun_student or row_id not in completed_prediction_ids
    ]

    if pending_student_indices:
        print(f"scoring missing rows: {len(pending_student_indices)} / {len(sample_df)}")
        student_cfg = deepcopy(cfg)
        student_cfg.brain_model_id = None
        student_cfg.grounding_model_id = None
        student_cfg.sam_model_id = None
        student_pipeline = MultiStageVQAPipeline(student_cfg)

        all_predictions = list(existing_rows)
        new_prediction_count = 0
        for batch_index in tqdm(pending_student_indices, desc=f"Student pass ({args.source})"):
            row = sample_df.iloc[batch_index]
            row_id = str(row["id"])
            context = context_map.get(row_id)
            if context is None:
                raise KeyError(f"missing cached context for {row_id}")

            choice_map, _ = build_choices(row, shuffle=False)
            policy_question_type, detector_policy = select_detector_policy(
                row["question"],
                choice_map,
                detector_policy_configs,
                planned_question_type=context.question_type,
            )
            original_config = apply_policy_to_config(student_cfg, detector_policy)
            try:
                prediction = student_pipeline.predict_row(row, context=context)
                brain_prior, brain_prior_weight = build_brain_choice_prior(context, choice_map, student_cfg)
                detect_choice_prior, detect_choice_prior_weight = build_detection_choice_prior(context, choice_map, student_cfg)
                count_prior, count_prior_weight = build_detection_count_prior(context, choice_map, student_cfg)
                (
                    detector_biased_probs,
                    detector_biased_answer,
                    detector_biased_confidence,
                    detector_blend_applied,
                    _,
                ) = build_detector_biased_probs(
                    prediction.aggregated_probs,
                    count_prior,
                    count_prior_weight,
                    detector_policy,
                    policy_question_type,
                )
            finally:
                restore_policy_to_config(student_cfg, original_config)

            final_answer = detector_biased_answer if detector_blend_applied else prediction.answer
            final_confidence = detector_biased_confidence if detector_blend_applied else prediction.confidence
            item = {
                "id": row_id,
                "path": row.get("path", ""),
                "question": row.get("question", ""),
                "a": row.get("a", ""),
                "b": row.get("b", ""),
                "c": row.get("c", ""),
                "d": row.get("d", ""),
                "answer": final_answer,
                "confidence": round(final_confidence, 6),
                "raw_answer": prediction.answer,
                "raw_confidence": round(prediction.confidence, 6),
                "question_type": context.question_type,
                "focus_objects": "|".join(context.focus_objects),
                "detector_blend_applied": detector_blend_applied,
                "brain_prior_weight": round(brain_prior_weight, 6),
                "detect_choice_prior_weight": round(detect_choice_prior_weight, 6),
                "count_prior_weight": round(count_prior_weight, 6),
            }
            item.update({f"raw_p_{label}": round(prediction.aggregated_probs[label], 6) for label in CHOICE_LABELS})
            item.update({f"final_p_{label}": round(detector_biased_probs[label], 6) for label in CHOICE_LABELS})
            for model_name, model_pred in prediction.model_predictions.items():
                item[f"{model_name}_answer"] = model_pred.answer
                item[f"{model_name}_confidence"] = round(model_pred.confidence, 6)
            all_predictions.append(item)

            new_prediction_count += 1
            if new_prediction_count % args.batch_save_every == 0:
                save_prediction_outputs(all_predictions, output_full_csv, output_answer_csv, sample_ids)

        result_df = save_prediction_outputs(all_predictions, output_full_csv, output_answer_csv, sample_ids)
        student_pipeline._release_context_models()
        del student_pipeline
        clear_runtime_memory()
    else:
        print(f"prediction csv already complete: {len(completed_prediction_ids)} / {len(sample_df)}")
        result_df = order_prediction_df(pd.DataFrame(existing_rows), sample_ids) if existing_rows else pd.DataFrame()

    print("saved full predictions:", output_full_csv)
    print("saved answer csv     :", output_answer_csv)
    print("saved rows           :", len(result_df))


if __name__ == "__main__":
    main()
