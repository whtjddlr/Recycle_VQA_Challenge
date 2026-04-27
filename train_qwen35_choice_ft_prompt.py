from __future__ import annotations
"""LoRA training and test inference entrypoint for the prompt branch.

This branch supports choice-aware loss computation and writes:
- test_predictions_detailed.csv
- submission.csv
"""

import argparse
import gc
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


Image.MAX_IMAGE_PIXELS = None

# Shared multiple-choice metadata used during both training and test inference.
CHOICE_LABELS = ["a", "b", "c", "d"]
DEFAULT_TTA_ORDERS = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [3, 2, 1, 0],
]

QUESTION_TYPE_RULES = [
    (
        "count",
        [
            "몇 개",
            "몇개",
            "개수",
            "갯수",
            "총 몇",
            "몇 병",
            "몇 캔",
            "몇 개인가",
            "몇 개입니까",
        ],
    ),
    ("color", ["색깔", "색상", "무슨 색", "어떤 색", "컬러"]),
    ("location", ["어디에", "어느 쪽", "위치는", "놓여 있", "어디에 있"]),
    ("dominant", ["가장 많이", "가장 많은", "주로", "대부분"]),
    ("recycle", ["재활용 분류", "분리배출", "배출 방법", "분리수거", "분류는", "분류해", "분류되는"]),
    ("material", ["재질", "소재", "재료", "만들어졌", "만들어진"]),
    (
        "state",
        [
            "비어",
            "찌그러",
            "구겨",
            "훼손",
            "오염",
            "라벨",
            "상태",
            "특징",
            "모양",
            "형태",
            "브랜드",
            "적힌",
            "문구",
            "용량",
            "맛인",
        ],
    ),
    ("type", ["무엇인가요", "무엇입니까", "어떤 것", "종류는"]),
]

SYSTEM_INSTRUCT = (
    "당신은 재활용품 이미지를 분석하는 전문가입니다. "
    "반드시 이미지에서 직접 보이는 정보만 사용하세요. "
    "질문의 대상, 재질, 종류, 상태와 상식적으로 맞지 않는 보기는 먼저 제외하세요. "
    "예를 들어 플라스틱을 묻는데 종이상자처럼 질문과 모순되거나, '플라스틱 캔'처럼 비현실적인 조합은 정답 후보에서 우선 배제하세요. "
    "작은 글씨, 숫자, 라벨, 인쇄 문구, 재질 표기가 중요하면 해당 부분을 더 가까이 확대해 본다고 생각하고 다시 확인하세요. "
    "정답은 a, b, c, d 중 정확히 하나의 소문자 한 글자로만 답하세요."
)

GLOBAL_PROMPT_HINTS = [
    "질문의 핵심 대상과 보기의 의미가 서로 맞는지 먼저 확인하고, 질문과 상식에 어긋나는 보기는 빠르게 제외하세요.",
    "예: 플라스틱을 묻는데 종이상자, 금속/캔을 묻는데 비닐봉지, 실제로 어색한 조합인 '플라스틱 캔' 같은 보기는 우선 배제하세요.",
    "라벨, 숫자, 글자, 재질 표기, 작은 부품이 중요하면 그 부분만 다시 크게 본다고 생각하고 확인하세요.",
    "대상이 작거나 멀리 있으면 주변 배경보다 대상 중심으로 다시 좁혀 본다고 생각하고 판단하세요.",
]

MATERIAL_CRITERIA_HINT = (
    "재질 기준: 플라스틱=반투명/불투명, 매끄러운 표면, 가벼워 보임, 구겨질 수 있음; "
    "유리=투명하고 광택 강함, 두꺼운 느낌; "
    "금속/캔=금속 광택, 알루미늄/철 캔 형태; "
    "종이/종이팩=불투명, 광택 적음, 코팅된 팩은 종이팩; "
    "비닐=얇고 투명/반투명, 구겨짐."
)

PROMPT_HINTS = {
    "count": [
        "질문 조건에 맞는 물체만 세고, 겹쳐 있어도 각각 따로 셉니다.",
        "색상, 크기, 재질이 다르면 별개로 보고 같은 물체를 중복해서 세지 마세요.",
    ],
    "color": [
        "질문 대상 물체의 실제 보이는 부분만 보고 색을 판단하세요.",
        "반사광, 그림자, 배경색은 무시하세요.",
    ],
    "material": [
        "표면 질감, 광택, 투명도, 형태를 보고 재질을 판단하세요.",
        MATERIAL_CRITERIA_HINT,
    ],
    "recycle": [
        "대상의 재질과 형태를 함께 보고 가장 적절한 분리배출 분류를 고르세요.",
        MATERIAL_CRITERIA_HINT,
    ],
    "dominant": [
        "보기 후보들을 빠르게 비교해 가장 많이 보이거나 가장 넓은 면적을 차지하는 쪽을 고르세요.",
    ],
    "location": [
        "대상을 먼저 찾고 주변 물체와의 상대적 위치로 판단하세요.",
    ],
    "state": [
        "대상 하나를 먼저 특정한 뒤 라벨, 찌그러짐, 오염, 비어 있음 같은 세부 속성만 집중해서 보세요.",
    ],
    "type": [
        "이미지에서 직접 보이는 물체의 종류를 확인하고 가장 구체적으로 맞는 보기를 고르세요.",
    ],
    "other": [
        "이미지를 먼저 보고 질문의 대상과 보기의 차이를 비교한 뒤 가장 맞는 보기를 고르세요.",
    ],
}


def format_seconds(total_seconds: Optional[float]) -> str:
    if total_seconds is None or not math.isfinite(total_seconds) or total_seconds < 0:
        return "?"
    rounded = int(round(float(total_seconds)))
    hours, rem = divmod(rounded, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def detect_default_device_for_args() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    default_device = detect_default_device_for_args()
    default_batch_size = 1 if default_device == "mps" else 8
    default_grad_accum = 8 if default_device == "mps" else 1
    default_num_workers = 0 if default_device == "mps" else 4
    default_min_pixels = 256 * 28 * 28 if default_device == "mps" else 512 * 28 * 28
    default_max_pixels = 1080 * 28 * 28 if default_device == "mps" else 1080 * 28 * 28
    default_quant_4bit = default_device == "cuda"
    default_train_monitor_samples = 64 if default_device == "mps" else 128

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5 for 4-choice VQA with answer-only supervision."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=Path("train.csv"))
    parser.add_argument("--valid-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "qwen35_4b_choice_ft")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.10)
    parser.add_argument("--debug-train-samples", type=int, default=0)
    parser.add_argument("--debug-valid-samples", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=default_min_pixels)
    parser.add_argument("--max-pixels", type=int, default=default_max_pixels)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--grad-accum", type=int, default=default_grad_accum)
    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["choice_ce", "generative"],
        default="choice_ce",
        help="Training loss: direct 4-way choice cross-entropy or the original generative loss.",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Stop after this many non-improving epochs on validation accuracy/loss.",
    )
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--quant-4bit",
        action=argparse.BooleanOptionalAction,
        default=default_quant_4bit,
        help="Use 4-bit QLoRA on CUDA. Default: enabled.",
    )
    parser.add_argument(
        "--shuffle-choices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle answer choices during training. Default: enabled.",
    )
    parser.add_argument(
        "--val-tta-orders",
        type=int,
        default=3,
        help="How many choice-order TTA patterns to use for validation accuracy.",
    )
    parser.add_argument(
        "--test-tta-orders",
        type=int,
        default=5,
        help="How many choice-order TTA patterns to use for final test inference.",
    )
    parser.add_argument(
        "--test-save-every",
        type=int,
        default=50,
        help="Autosave test inference outputs every N samples. 0 disables partial saves.",
    )
    parser.add_argument(
        "--save-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save adapter/processor checkpoints after every epoch. Default: enabled.",
    )
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-monitor-samples",
        type=int,
        default=default_train_monitor_samples,
        help="How many train samples to score each epoch to watch for overfitting. 0 disables it.",
    )
    parser.add_argument(
        "--overfit-gap-threshold",
        type=float,
        default=0.08,
        help="Warn if monitored train accuracy exceeds validation accuracy by at least this margin.",
    )
    parser.add_argument(
        "--train-extra-context-col",
        type=str,
        default=None,
        help="Optional text column to append as extra context during training/validation prompts.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_runtime_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def get_autocast_context(device: str) -> Any:
    if device != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def resolve_path(project_root: Path, path: Path | str | None) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def detect_qtype(question: str, choice_map: Optional[Dict[str, str]] = None) -> str:
    question = str(question)
    for question_type, keywords in QUESTION_TYPE_RULES:
        if any(keyword in question for keyword in keywords):
            return question_type
    if choice_map:
        numeric_choices = 0
        for label in CHOICE_LABELS:
            choice_text = str(choice_map.get(label, "")).strip()
            if any(ch.isdigit() for ch in choice_text) or choice_text.endswith(("개", "명", "병", "캔")):
                numeric_choices += 1
        if numeric_choices >= 3:
            return "count"
    return "basic"


def get_system_instruction(question: str, choice_map: Optional[Dict[str, str]] = None) -> str:
    return SYSTEM_INSTRUCT


def get_prompt_hint(question: str, choice_map: Dict[str, str]) -> str:
    qtype = detect_qtype(question, choice_map)
    hints = PROMPT_HINTS.get(qtype, PROMPT_HINTS["other"])
    return "\n".join(f"- {hint}" for hint in hints if hint)


def build_choice_map(row: pd.Series) -> Dict[str, str]:
    return {label: str(row.get(label, "") or "") for label in CHOICE_LABELS}


def build_mc_prompt(
    question: str,
    choices_ordered: Sequence[Tuple[str, str]],
    extra_context: str = "",
) -> str:
    choice_map = {label: text for label, text in choices_ordered}
    lines = [f"질문: {str(question).strip()}"]
    if GLOBAL_PROMPT_HINTS:
        lines.extend(["", "공통 원칙:"])
        lines.extend(f"- {hint}" for hint in GLOBAL_PROMPT_HINTS if hint)
    hint = get_prompt_hint(question, choice_map)
    if hint:
        lines.extend(["", "유형별 판단 기준:", hint])
    if extra_context.strip():
        lines.extend(["", "[참고 맥락]", extra_context.strip()])
    lines.append("")
    for label, text in choices_ordered:
        lines.append(f"({label}) {text}")
    lines.extend(["", "정답:"])
    return "\n".join(lines)


def safe_json_loads(raw: Any) -> Any:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text in {"nan", "None", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def build_retrieval_context_from_row(row: pd.Series, max_hits: int = 3) -> str:
    hits = safe_json_loads(row.get("retrieval_context_json"))
    if not isinstance(hits, list) or not hits:
        return ""

    lines: List[str] = ["[retrieval 참고]"]

    consensus_label = str(row.get("retrieval_consensus_label", "") or "").strip().lower()
    consensus_text = str(row.get("retrieval_consensus_text", "") or "").strip()
    if consensus_label in CHOICE_LABELS:
        if consensus_text:
            lines.append(f"retrieval 추천: ({consensus_label}) {consensus_text}")
        else:
            lines.append(f"retrieval 추천: ({consensus_label})")

    for hit in hits[:max_hits]:
        if not isinstance(hit, dict):
            continue
        split = str(hit.get("ref_split", "") or "")
        label = str(hit.get("ref_answer_label", "") or "").strip().lower()
        answer_text = str(hit.get("ref_answer_text", "") or "").strip()
        same_q = bool(hit.get("same_question_exact", False))
        exact_img = bool(hit.get("exact_hash_match", False))
        image_sim = float(hit.get("image_similarity", 0.0) or 0.0)
        ssim = float(hit.get("ssim", 0.0) or 0.0)
        prefix = f"- {split} hit"
        details = [
            f"same_q={same_q}",
            f"exact_img={exact_img}",
            f"clip={image_sim:.3f}",
        ]
        if ssim > 0.0:
            details.append(f"ssim={ssim:.3f}")
        answer_part = f"answer=({label}) {answer_text}".strip()
        lines.append(f"{prefix}: {', '.join(details)}, {answer_part}")

    lines.append("참고 정보가 현재 이미지와 다르면 현재 이미지와 보기 판단을 우선하세요.")
    return "\n".join(lines)


def build_dino_context_from_row(row: pd.Series) -> str:
    focus_objects = str(row.get("dino_focus_objects", "") or "").strip()
    choice_prior = safe_json_loads(row.get("dino_choice_prior_json"))
    count_prior = safe_json_loads(row.get("dino_count_prior_json"))
    grounding_quality = float(row.get("dino_grounding_quality", 0.0) or 0.0)

    lines: List[str] = []
    if focus_objects:
        focus_items = [item for item in focus_objects.split("|") if item]
        if focus_items:
            lines.append("[dino 참고]")
            lines.append("주목 대상: " + ", ".join(focus_items[:8]))

    if isinstance(choice_prior, dict) and choice_prior:
        score_text = ", ".join(
            f"{label}={float(choice_prior.get(label, 0.0)):.2f}"
            for label in CHOICE_LABELS
        )
        if not lines:
            lines.append("[dino 참고]")
        lines.append("선택지 힌트: " + score_text)

    if isinstance(count_prior, dict) and count_prior:
        score_text = ", ".join(
            f"{label}={float(count_prior.get(label, 0.0)):.2f}"
            for label in CHOICE_LABELS
        )
        if not lines:
            lines.append("[dino 참고]")
        lines.append("개수 힌트: " + score_text)

    if grounding_quality > 0.0:
        if not lines:
            lines.append("[dino 참고]")
        lines.append(f"검출 품질: {grounding_quality:.2f}")

    return "\n".join(lines)


def build_row_extra_context(row: pd.Series, explicit_col: Optional[str] = None) -> str:
    parts: List[str] = []
    if explicit_col and explicit_col in row.index:
        explicit_text = str(row.get(explicit_col, "") or "").strip()
        if explicit_text:
            parts.append(explicit_text)

    retrieval_context = build_retrieval_context_from_row(row)
    if retrieval_context:
        parts.append(retrieval_context)

    dino_context = build_dino_context_from_row(row)
    if dino_context:
        parts.append(dino_context)

    return "\n\n".join(part for part in parts if part).strip()


def load_image(path: Path | str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resolve_image_path(row: pd.Series, image_root: Path) -> str:
    raw = str(row.get("path", "") or "").strip()
    candidates: List[Path] = []
    if raw:
        raw_path = Path(raw)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        candidates.append(image_root / raw)
        candidates.append(image_root / raw_path.name)
    row_id = str(row.get("id", "") or "").strip()
    if row_id:
        candidates.append(image_root / row_id)
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            candidates.append(image_root / f"{row_id}{ext}")

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Could not resolve image path for row id={row.get('id')} path={raw!r}")


def load_dataframe(csv_path: Path, image_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "id" not in df.columns:
        df["id"] = [str(index) for index in range(len(df))]
    df["path"] = df.apply(lambda row: resolve_image_path(row, image_root), axis=1)
    return df


def add_question_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["qtype"] = df.apply(
        lambda row: detect_qtype(str(row.get("question", "")), build_choice_map(row)),
        axis=1,
    )
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.lower()
    return df


def split_train_valid(
    train_df: pd.DataFrame,
    valid_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify_label = train_df["qtype"].astype(str) + "__" + train_df["answer"].astype(str)
    label_counts = stratify_label.value_counts()
    rare_labels = set(label_counts[label_counts < 2].index.tolist())
    if rare_labels:
        stratify_values = train_df["answer"].astype(str)
    else:
        stratify_values = stratify_label

    train_subset, valid_subset = train_test_split(
        train_df,
        test_size=valid_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_values if stratify_values.nunique() > 1 else None,
    )
    return train_subset.reset_index(drop=True), valid_subset.reset_index(drop=True)


def apply_chat_template_no_think(
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = False,
) -> Any:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            return processor.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )


class Qwen35ChoiceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        train: bool,
        shuffle_choices: bool,
        extra_context_col: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.shuffle_choices = bool(shuffle_choices and train)
        self.extra_context_col = extra_context_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image = load_image(row["path"])

        orig_choices = [(label, str(row[label])) for label in CHOICE_LABELS]
        if self.shuffle_choices:
            indices = list(range(len(orig_choices)))
            random.shuffle(indices)
            displayed_choices = [("abcd"[new_idx], orig_choices[old_idx][1]) for new_idx, old_idx in enumerate(indices)]
            answer_map = {orig_choices[old_idx][0]: "abcd"[new_idx] for new_idx, old_idx in enumerate(indices)}
        else:
            displayed_choices = orig_choices
            answer_map = {label: label for label in CHOICE_LABELS}

        extra_context = build_row_extra_context(row, self.extra_context_col)

        question = str(row["question"])
        prompt_text = build_mc_prompt(question, displayed_choices, extra_context=extra_context)
        system_instruction = get_system_instruction(question, {k: v for k, v in displayed_choices})
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        item: Dict[str, Any] = {
            "id": str(row["id"]),
            "path": str(row["path"]),
            "image": image,
            "question": question,
            "prompt_messages": prompt_messages,
            "prompt_text": prompt_text,
            "displayed_choices": displayed_choices,
            "qtype": str(row.get("qtype", "basic")),
        }
        if self.train:
            answer = str(row["answer"]).strip().lower()
            remapped_answer = answer_map.get(answer, answer)
            item["answer"] = remapped_answer
            item["full_messages"] = prompt_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": remapped_answer}]}
            ]
        return item


@dataclass
class TrainCollator:
    processor: Any
    loss_mode: str = "choice_ce"

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images = [sample["image"] for sample in batch]
        prompt_texts = [
            apply_chat_template_no_think(
                self.processor,
                sample["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for sample in batch
        ]
        prompt_enc = self.processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")

        if self.loss_mode == "choice_ce":
            choice_labels = torch.tensor(
                [CHOICE_LABELS.index(str(sample["answer"]).strip().lower()) for sample in batch],
                dtype=torch.long,
            )
            prompt_enc["choice_labels"] = choice_labels
            return prompt_enc

        full_texts = [
            apply_chat_template_no_think(
                self.processor,
                sample["full_messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for sample in batch
        ]
        full_enc = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")

        labels = full_enc["input_ids"].clone()
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lens):
            labels[row_idx, :prompt_len] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        full_enc["labels"] = labels
        return full_enc


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def resolve_choice_token_ids(processor: Any) -> Dict[str, int]:
    tokenizer = processor.tokenizer
    token_ids: Dict[str, int] = {}
    for label in CHOICE_LABELS:
        candidates = [
            tokenizer.encode(label, add_special_tokens=False),
            tokenizer.encode(f" {label}", add_special_tokens=False),
            tokenizer.encode(f"\n{label}", add_special_tokens=False),
        ]
        token_id = next((tokens[0] for tokens in candidates if len(tokens) == 1), None)
        if token_id is None:
            raise ValueError(f"Could not resolve single-token id for choice label {label!r}.")
        token_ids[label] = int(token_id)
    return token_ids


def build_choice_token_id_tensor(choice_token_ids: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([choice_token_ids[label] for label in CHOICE_LABELS], dtype=torch.long)


def compute_choice_ce_loss(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_token_id_tensor: torch.Tensor,
    choice_labels: torch.Tensor,
) -> torch.Tensor:
    if choice_token_id_tensor.device != logits.device:
        choice_token_id_tensor = choice_token_id_tensor.to(logits.device)

    seq_lens = attention_mask.sum(dim=1).long().sub(1).clamp_min(0)
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    next_token_logits = logits[batch_indices, seq_lens, :]
    choice_logits = next_token_logits.index_select(dim=-1, index=choice_token_id_tensor)
    return F.cross_entropy(choice_logits, choice_labels.to(choice_logits.device))


def forward_with_loss(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    loss_mode: str,
    choice_token_id_tensor: Optional[torch.Tensor] = None,
) -> Tuple[Any, torch.Tensor]:
    if loss_mode == "choice_ce":
        if choice_token_id_tensor is None:
            raise ValueError("choice_token_id_tensor is required when loss_mode='choice_ce'.")
        model_inputs = {key: value for key, value in batch.items() if key != "choice_labels"}
        outputs = model(**model_inputs)
        loss = compute_choice_ce_loss(
            outputs.logits,
            model_inputs["attention_mask"],
            choice_token_id_tensor,
            batch["choice_labels"],
        )
        return outputs, loss

    outputs = model(**batch)
    return outputs, outputs.loss


def build_validation_orders(num_orders: int) -> List[List[int]]:
    # Test-time augmentation is implemented by rotating the displayed choice order.
    num_orders = max(1, min(num_orders, len(DEFAULT_TTA_ORDERS)))
    return [order[:] for order in DEFAULT_TTA_ORDERS[:num_orders]]


def predict_choice_details_from_row(
    row: pd.Series,
    model: torch.nn.Module,
    processor: Any,
    device: str,
    choice_token_ids: Dict[str, int],
    *,
    num_tta_orders: int = 1,
    extra_context_col: Optional[str] = None,
) -> Dict[str, Any]:
    # Score one sample over several choice-order permutations, then aggregate the
    # per-order probabilities into a single detailed prediction record.
    image = load_image(row["path"])
    question = str(row["question"])
    orig_texts = [str(row[label]) for label in CHOICE_LABELS]
    extra_context = build_row_extra_context(row, extra_context_col)

    votes: List[str] = []
    winner_scores: List[float] = []
    prob_sums = {label: 0.0 for label in CHOICE_LABELS}
    logit_sums = {label: 0.0 for label in CHOICE_LABELS}
    vote_counts = {label: 0 for label in CHOICE_LABELS}
    tta_records: List[Dict[str, Any]] = []

    try:
        for order in build_validation_orders(num_tta_orders):
            displayed_choices = [("abcd"[new_idx], orig_texts[old_idx]) for new_idx, old_idx in enumerate(order)]
            display_to_orig = {"abcd"[new_idx]: CHOICE_LABELS[old_idx] for new_idx, old_idx in enumerate(order)}
            prompt_text = build_mc_prompt(question, displayed_choices, extra_context=extra_context)
            system_instruction = get_system_instruction(question, {k: v for k, v in displayed_choices})

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            chat_text = apply_chat_template_no_think(
                processor,
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[chat_text], images=[image], return_tensors="pt")
            inputs = move_batch_to_device(inputs, device)

            with torch.no_grad():
                with get_autocast_context(device):
                    outputs = model(**inputs)

            # Multimodal prompt packing can yield shorter logits than the raw
            # attention length, so clamp to the last valid logits position.
            last_pos = min(
                int(inputs["attention_mask"].sum(dim=1).item() - 1),
                int(outputs.logits.shape[1] - 1),
            )
            next_token_logits = outputs.logits[0, last_pos].detach().float().cpu()
            raw_scores = {
                label: float(next_token_logits[token_id].item())
                for label, token_id in choice_token_ids.items()
            }
            score_tensor = torch.tensor([raw_scores[label] for label in CHOICE_LABELS], dtype=torch.float32)
            probs_tensor = torch.softmax(score_tensor, dim=0)
            pred_display = CHOICE_LABELS[int(torch.argmax(probs_tensor).item())]
            pred_orig = display_to_orig[pred_display]
            confidence = float(torch.max(probs_tensor).item())
            votes.append(pred_orig)
            winner_scores.append(confidence)
            vote_counts[pred_orig] += 1

            order_probs: Dict[str, float] = {}
            order_logits: Dict[str, float] = {}
            for display_idx, display_label in enumerate(CHOICE_LABELS):
                orig_label = display_to_orig[display_label]
                order_probs[orig_label] = float(probs_tensor[display_idx].item())
                order_logits[orig_label] = float(raw_scores[display_label])
            for label in CHOICE_LABELS:
                prob_sums[label] += order_probs[label]
                logit_sums[label] += order_logits[label]
            tta_records.append(
                {
                    "order": "".join(CHOICE_LABELS[old_idx] for old_idx in order),
                    "pred": pred_orig,
                    "confidence": confidence,
                    "probabilities": {label: round(order_probs[label], 6) for label in CHOICE_LABELS},
                }
            )
    finally:
        image.close()

    if not votes:
        return {
            "pred": "a",
            "confidence": 0.0,
            "margin_top2": 0.0,
            "entropy": 0.0,
            "vote_count_a": 0,
            "vote_count_b": 0,
            "vote_count_c": 0,
            "vote_count_d": 0,
            "avg_logit_a": 0.0,
            "avg_logit_b": 0.0,
            "avg_logit_c": 0.0,
            "avg_logit_d": 0.0,
            "avg_p_a": 0.0,
            "avg_p_b": 0.0,
            "avg_p_c": 0.0,
            "avg_p_d": 0.0,
            "tta_votes_json": "[]",
        }

    final_label = max(CHOICE_LABELS, key=lambda label: (votes.count(label), -CHOICE_LABELS.index(label)))
    kept_scores = [score for vote, score in zip(votes, winner_scores) if vote == final_label]
    num_votes = len(votes)
    avg_probs = {label: float(prob_sums[label] / max(1, num_votes)) for label in CHOICE_LABELS}
    avg_logits = {label: float(logit_sums[label] / max(1, num_votes)) for label in CHOICE_LABELS}
    sorted_probs = sorted(avg_probs.values(), reverse=True)
    margin_top2 = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 0.0
    entropy = float(-sum(prob * math.log(max(prob, 1e-12)) for prob in avg_probs.values()))
    avg_conf = float(avg_probs[final_label])
    if kept_scores:
        avg_conf = float((avg_conf + (sum(kept_scores) / len(kept_scores))) / 2.0)
    return {
        "pred": final_label,
        "confidence": avg_conf,
        "margin_top2": margin_top2,
        "entropy": entropy,
        "vote_count_a": int(vote_counts["a"]),
        "vote_count_b": int(vote_counts["b"]),
        "vote_count_c": int(vote_counts["c"]),
        "vote_count_d": int(vote_counts["d"]),
        "avg_logit_a": avg_logits["a"],
        "avg_logit_b": avg_logits["b"],
        "avg_logit_c": avg_logits["c"],
        "avg_logit_d": avg_logits["d"],
        "avg_p_a": avg_probs["a"],
        "avg_p_b": avg_probs["b"],
        "avg_p_c": avg_probs["c"],
        "avg_p_d": avg_probs["d"],
        "tta_votes_json": json.dumps(tta_records, ensure_ascii=False),
    }


def predict_choice_from_row(
    row: pd.Series,
    model: torch.nn.Module,
    processor: Any,
    device: str,
    choice_token_ids: Dict[str, int],
    *,
    num_tta_orders: int = 1,
    extra_context_col: Optional[str] = None,
) -> Tuple[str, float]:
    details = predict_choice_details_from_row(
        row,
        model,
        processor,
        device,
        choice_token_ids,
        num_tta_orders=num_tta_orders,
        extra_context_col=extra_context_col,
    )
    return str(details["pred"]), float(details["confidence"])


def predict_dataframe_details(
    df: pd.DataFrame,
    model: torch.nn.Module,
    processor: Any,
    device: str,
    choice_token_ids: Dict[str, int],
    *,
    num_tta_orders: int,
    extra_context_col: Optional[str],
    progress_desc: str,
    save_every: int = 0,
    save_callback: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    save_every = max(0, int(save_every or 0))
    for idx in tqdm(range(len(df)), desc=progress_desc, unit="sample", leave=True, dynamic_ncols=True):
        record = predict_choice_details_from_row(
            df.iloc[idx],
            model,
            processor,
            device,
            choice_token_ids,
            num_tta_orders=num_tta_orders,
            extra_context_col=extra_context_col,
        )
        records.append(record)
        if save_callback is not None and save_every > 0 and (idx + 1) % save_every == 0:
            save_callback(pd.DataFrame(records))
    if save_callback is not None and records and save_every > 0 and len(records) % save_every != 0:
        save_callback(pd.DataFrame(records))
    return pd.DataFrame(records)


def write_test_inference_outputs(
    *,
    base_df: pd.DataFrame,
    pred_details: pd.DataFrame,
    detailed_path: Path,
    submission_path: Path,
    summary_path: Path,
    test_csv: Path,
    tta_orders: int,
    elapsed_seconds: float,
    partial: bool,
) -> None:
    aligned_base = base_df.iloc[: len(pred_details)].copy().reset_index(drop=True)
    aligned_preds = pred_details.reset_index(drop=True)
    for column in aligned_preds.columns:
        aligned_base[column] = aligned_preds[column]
    detailed_df = aligned_base.rename(columns={"pred": "answer"})
    detailed_df["answer_text"] = detailed_df.apply(
        lambda row: str(row.get(str(row["answer"]).strip().lower(), "")),
        axis=1,
    )
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    submission_df = detailed_df[["id", "answer"]].copy()
    submission_df.to_csv(submission_path, index=False, encoding="utf-8-sig")

    summary_payload = {
        "test_csv": str(test_csv),
        "num_samples": int(len(base_df)),
        "num_saved_samples": int(len(detailed_df)),
        "tta_orders": int(tta_orders),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "detailed_predictions_path": str(detailed_path),
        "submission_path": str(submission_path),
        "partial": bool(partial),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_qtype_accuracy(valid_df: pd.DataFrame, preds: Sequence[str]) -> pd.Series:
    tmp = valid_df.copy()
    tmp["pred"] = [str(pred).strip().lower() for pred in preds]
    tmp["gold"] = tmp["answer"].astype(str).str.strip().str.lower()
    tmp["correct"] = (tmp["pred"] == tmp["gold"]).astype(int)
    if "qtype" not in tmp.columns:
        tmp["qtype"] = tmp.apply(
            lambda row: detect_qtype(str(row.get("question", "")), build_choice_map(row)),
            axis=1,
        )
    return tmp.groupby("qtype")["correct"].mean().sort_index()


def build_model_load_kwargs(device: str, quant_4bit: bool) -> Dict[str, Any]:
    dtype = get_runtime_dtype(device)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if quant_4bit and device == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["device_map"] = "auto"
    else:
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def load_qwen35_lora_model(
    model_id: str,
    device: str,
    min_pixels: int,
    max_pixels: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    quant_4bit: bool,
) -> Tuple[torch.nn.Module, Any]:
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        **build_model_load_kwargs(device, quant_4bit),
    )
    if quant_4bit and device == "cuda":
        base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    if device != "cuda":
        model = model.to(device)
    return model, processor


def load_qwen35_adapter_for_inference(
    model_id: str,
    adapter_dir: Path,
    device: str,
    min_pixels: int,
    max_pixels: int,
    quant_4bit: bool,
) -> Tuple[torch.nn.Module, Any]:
    processor_source = adapter_dir if adapter_dir.exists() else model_id
    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        **build_model_load_kwargs(device, quant_4bit),
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    if device != "cuda":
        model = model.to(device)
    return model, processor


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    best_val_acc: float,
    best_val_loss: float,
) -> None:
    payload = {
        "model_id": args.model_id,
        "seed": args.seed,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "num_epochs": int(args.num_epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "best_val_acc": round(float(best_val_acc), 6),
        "best_val_loss": round(float(best_val_loss), 6),
        "loss_mode": args.loss_mode,
        "shuffle_choices": bool(args.shuffle_choices),
        "val_tta_orders": int(args.val_tta_orders),
        "min_pixels": int(args.min_pixels),
        "max_pixels": int(args.max_pixels),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_epoch_checkpoint(
    model: torch.nn.Module,
    processor: Any,
    epoch_dir: Path,
    *,
    epoch: int,
    val_acc: float,
    val_loss: float,
    global_step: int,
) -> None:
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)
    (epoch_dir / "metrics.json").write_text(
        json.dumps(
            {
                "epoch": int(epoch),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "global_step": int(global_step),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_dataframe_accuracy(
    df: pd.DataFrame,
    model: torch.nn.Module,
    processor: Any,
    device: str,
    choice_token_ids: Dict[str, int],
    *,
    num_tta_orders: int,
    extra_context_col: Optional[str],
    progress_desc: str = "score",
) -> Tuple[float, List[str], List[float]]:
    pred_details = predict_dataframe_details(
        df,
        model,
        processor,
        device,
        choice_token_ids,
        num_tta_orders=num_tta_orders,
        extra_context_col=extra_context_col,
        progress_desc=progress_desc,
    )
    preds = pred_details["pred"].astype(str).tolist()
    confidences = pred_details["confidence"].astype(float).tolist()

    golds = df["answer"].astype(str).str.strip().str.lower().tolist()
    correct = sum(int(pred == gold) for pred, gold in zip(preds, golds))
    acc = correct / max(1, len(golds))
    return acc, preds, confidences


def run_test_inference(
    *,
    test_csv: Path,
    image_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    use_4bit: bool,
) -> None:
    print("\n[test inference] loading best adapter for test.csv...", flush=True)
    best_dir = output_dir / "best_adapter"
    if not best_dir.exists():
        raise FileNotFoundError(f"Missing best adapter directory: {best_dir}")

    test_df = add_question_metadata(load_dataframe(test_csv, image_root))
    print(f"[test inference] test size={len(test_df)} | tta_orders={args.test_tta_orders}", flush=True)

    model, processor = load_qwen35_adapter_for_inference(
        model_id=args.model_id,
        adapter_dir=best_dir,
        device=device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        quant_4bit=use_4bit,
    )
    choice_token_ids = resolve_choice_token_ids(processor)

    detailed_path = output_dir / "test_predictions_detailed.csv"
    submission_path = output_dir / "submission.csv"
    summary_path = output_dir / "test_inference_summary.json"

    inference_start = time.perf_counter()

    def save_partial_outputs(pred_snapshot: pd.DataFrame) -> None:
        write_test_inference_outputs(
            base_df=test_df,
            pred_details=pred_snapshot,
            detailed_path=detailed_path,
            submission_path=submission_path,
            summary_path=summary_path,
            test_csv=test_csv,
            tta_orders=args.test_tta_orders,
            elapsed_seconds=time.perf_counter() - inference_start,
            partial=True,
        )
        print(
            f"[test inference] autosaved {len(pred_snapshot)}/{len(test_df)} -> "
            f"{detailed_path.name}, {submission_path.name}",
            flush=True,
        )

    pred_details = predict_dataframe_details(
        test_df,
        model,
        processor,
        device,
        choice_token_ids,
        num_tta_orders=args.test_tta_orders,
        extra_context_col=args.train_extra_context_col,
        progress_desc="test-infer",
        save_every=args.test_save_every,
        save_callback=save_partial_outputs if int(args.test_save_every or 0) > 0 else None,
    )
    inference_elapsed = time.perf_counter() - inference_start

    write_test_inference_outputs(
        base_df=test_df,
        pred_details=pred_details,
        detailed_path=detailed_path,
        submission_path=submission_path,
        summary_path=summary_path,
        test_csv=test_csv,
        tta_orders=args.test_tta_orders,
        elapsed_seconds=inference_elapsed,
        partial=False,
    )
    print(
        f"[test inference] done in {format_seconds(inference_elapsed)} -> "
        f"{detailed_path.name}, {submission_path.name}",
        flush=True,
    )
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    train_csv = resolve_path(project_root, args.train_csv)
    valid_csv = resolve_path(project_root, args.valid_csv)
    test_csv = resolve_path(project_root, args.test_csv)
    image_root = resolve_path(project_root, args.image_root) or project_root
    output_dir = resolve_path(project_root, args.output_dir) or (project_root / "artifacts" / "qwen35_choice_ft")
    best_dir = output_dir / "best_adapter"
    last_dir = output_dir / "last_adapter"
    epoch_ckpt_root = output_dir / "epochs"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)
    epoch_ckpt_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_default_device()
    use_4bit = bool(args.quant_4bit and device == "cuda")

    print(f"device={device} | use_4bit={use_4bit}", flush=True)
    print(f"train_csv={train_csv}", flush=True)
    if valid_csv:
        print(f"valid_csv={valid_csv}", flush=True)
    if test_csv:
        print(f"test_csv={test_csv}", flush=True)

    # The same script supports full training and inference-only mode. If
    # num_epochs is 0 but test_csv is given, it still loads best_adapter and
    # writes test predictions.
    train_df = add_question_metadata(load_dataframe(train_csv, image_root))
    if args.debug_train_samples > 0:
        train_df = train_df.sample(
            n=min(args.debug_train_samples, len(train_df)),
            random_state=args.seed,
        ).reset_index(drop=True)

    if valid_csv:
        valid_df = add_question_metadata(load_dataframe(valid_csv, image_root))
    else:
        train_df, valid_df = split_train_valid(train_df, args.valid_size, args.seed)

    if args.debug_valid_samples > 0:
        valid_df = valid_df.iloc[: args.debug_valid_samples].reset_index(drop=True)

    print(f"train size={len(train_df)}", flush=True)
    print(f"valid size={len(valid_df)}", flush=True)
    print("\n[train qtype distribution]", flush=True)
    print(train_df["qtype"].value_counts(dropna=False), flush=True)
    print("\n[valid qtype distribution]", flush=True)
    print(valid_df["qtype"].value_counts(dropna=False), flush=True)

    train_monitor_df = pd.DataFrame(columns=train_df.columns)
    if args.train_monitor_samples > 0:
        train_monitor_df = train_df.sample(
            n=min(args.train_monitor_samples, len(train_df)),
            random_state=args.seed,
        ).reset_index(drop=True)
        print(f"\n[train monitor subset] {len(train_monitor_df)} samples", flush=True)

    model, processor = load_qwen35_lora_model(
        model_id=args.model_id,
        device=device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        quant_4bit=use_4bit,
    )
    model.print_trainable_parameters()

    train_ds = Qwen35ChoiceDataset(
        train_df,
        train=True,
        shuffle_choices=args.shuffle_choices,
        extra_context_col=args.train_extra_context_col,
    )
    valid_ds = Qwen35ChoiceDataset(
        valid_df,
        train=True,
        shuffle_choices=False,
        extra_context_col=args.train_extra_context_col,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TrainCollator(processor, loss_mode=args.loss_mode),
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=bool(args.num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TrainCollator(processor, loss_mode=args.loss_mode),
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=bool(device == "cuda"),
    )
    num_update_steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    num_training_steps = max(1, args.num_epochs * num_update_steps_per_epoch)
    warmup_steps = max(1, int(num_training_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    use_bf16 = bool(device == "cuda" and torch.cuda.is_bf16_supported())
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not use_bf16))
    choice_token_ids = resolve_choice_token_ids(processor)
    choice_token_id_tensor = build_choice_token_id_tensor(choice_token_ids)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    global_step = 0
    stale_epochs = 0
    metrics_history: List[Dict[str, Any]] = []
    run_start_time = time.perf_counter()

    print(
        "\n[run config]\n"
        f"model_id={args.model_id}\n"
        f"loss_mode={args.loss_mode}\n"
        f"batch_size={args.batch_size} | grad_accum={args.grad_accum} | "
        f"train_batches={len(train_loader)} | valid_batches={len(valid_loader)}\n"
        f"updates_per_epoch={num_update_steps_per_epoch} | total_updates={num_training_steps} | "
        f"val_tta_orders={args.val_tta_orders}\n"
        f"tqdm bars show [elapsed<remaining] while running.",
        flush=True,
    )

    for epoch in range(args.num_epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        accum_steps = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs} [train]",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_bar, start=1):
            batch = move_batch_to_device(batch, device)
            with get_autocast_context(device):
                outputs, loss = forward_with_loss(
                    model,
                    batch,
                    loss_mode=args.loss_mode,
                    choice_token_id_tensor=choice_token_id_tensor,
                )
                loss = loss / max(1, args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item())
            accum_steps += 1

            if step % args.grad_accum == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                avg_loss = running_loss / max(1, accum_steps)
                train_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "step": global_step,
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
                running_loss = 0.0
                accum_steps = 0

        model.eval()
        val_loss_total = 0.0
        val_loss_steps = 0
        with torch.no_grad():
            for batch in tqdm(
                valid_loader,
                desc=f"Epoch {epoch + 1}/{args.num_epochs} [valid-loss]",
                unit="batch",
                dynamic_ncols=True,
            ):
                batch = move_batch_to_device(batch, device)
                with get_autocast_context(device):
                    outputs, loss = forward_with_loss(
                        model,
                        batch,
                        loss_mode=args.loss_mode,
                        choice_token_id_tensor=choice_token_id_tensor,
                    )
                val_loss_total += float(loss.item())
                val_loss_steps += 1
        val_loss = val_loss_total / max(1, val_loss_steps)

        print(f"\nEpoch {epoch + 1}/{args.num_epochs} validation scoring...", flush=True)
        valid_pred_details = predict_dataframe_details(
            valid_df,
            model,
            processor,
            device,
            choice_token_ids,
            num_tta_orders=args.val_tta_orders,
            extra_context_col=args.train_extra_context_col,
            progress_desc=f"Epoch {epoch + 1}/{args.num_epochs} [valid-detail]",
        )
        preds = valid_pred_details["pred"].astype(str).tolist()
        golds = valid_df["answer"].astype(str).str.strip().str.lower().tolist()
        val_acc = sum(int(pred == gold) for pred, gold in zip(preds, golds)) / max(1, len(golds))
        qtype_acc = compute_qtype_accuracy(valid_df, preds)

        train_monitor_acc = None
        if not train_monitor_df.empty:
            print(f"Epoch {epoch + 1}/{args.num_epochs} train-monitor scoring...", flush=True)
            train_monitor_acc, _, _ = evaluate_dataframe_accuracy(
                train_monitor_df,
                model,
                processor,
                device,
                choice_token_ids,
                num_tta_orders=1,
                extra_context_col=args.train_extra_context_col,
                progress_desc=f"Epoch {epoch + 1}/{args.num_epochs} [train-monitor]",
            )

        epoch_elapsed = time.perf_counter() - epoch_start_time
        total_elapsed = time.perf_counter() - run_start_time
        avg_epoch_elapsed = total_elapsed / max(1, epoch + 1)
        est_remaining = avg_epoch_elapsed * max(0, args.num_epochs - (epoch + 1))

        print(
            f"\n[Epoch {epoch + 1}] val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"epoch_time={format_seconds(epoch_elapsed)} | total_elapsed={format_seconds(total_elapsed)} | "
            f"est_remaining={format_seconds(est_remaining)}",
            flush=True,
        )
        if train_monitor_acc is not None:
            print(f"[Epoch {epoch + 1}] train_monitor_acc={train_monitor_acc:.4f}", flush=True)
            if train_monitor_acc - val_acc >= args.overfit_gap_threshold:
                print(
                    f"[Warning] overfitting suspicion: train_monitor_acc - val_acc = "
                    f"{train_monitor_acc - val_acc:.4f}",
                    flush=True,
                )
        print("\n[qtype accuracy]", flush=True)
        print(qtype_acc, flush=True)

        valid_pred_df = valid_df[["id", "path", "question", "a", "b", "c", "d", "answer", "qtype"]].copy()
        for column in valid_pred_details.columns:
            valid_pred_df[column] = valid_pred_details[column]
        valid_pred_df["gold_text"] = valid_pred_df.apply(
            lambda row: str(row.get(str(row["answer"]).strip().lower(), "")),
            axis=1,
        )
        valid_pred_df["pred_text"] = valid_pred_df.apply(
            lambda row: str(row.get(str(row["pred"]).strip().lower(), "")),
            axis=1,
        )
        valid_pred_df["correct"] = valid_pred_df["answer"] == valid_pred_df["pred"]
        valid_pred_path = output_dir / f"valid_predictions_epoch{epoch + 1}.csv"
        valid_pred_df.to_csv(valid_pred_path, index=False, encoding="utf-8-sig")
        metrics_history.append(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_monitor_acc": train_monitor_acc,
            }
        )
        pd.DataFrame(metrics_history).to_csv(
            output_dir / "metrics_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if args.save_each_epoch:
            epoch_dir = epoch_ckpt_root / f"epoch_{epoch + 1:02d}"
            save_epoch_checkpoint(
                model,
                processor,
                epoch_dir,
                epoch=epoch + 1,
                val_acc=val_acc,
                val_loss=val_loss,
                global_step=global_step,
            )
            print(f"Epoch checkpoint saved -> {epoch_dir}", flush=True)

        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (best_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "global_step": global_step,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Best adapter saved -> {best_dir}", flush=True)
            stale_epochs = 0
        else:
            stale_epochs += 1
            print(
                f"No validation improvement for {stale_epochs} epoch(s) "
                f"(patience={args.early_stopping_patience}).",
                flush=True,
            )
            if stale_epochs >= args.early_stopping_patience:
                print("Early stopping triggered.", flush=True)
                break

    if args.save_last:
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)
        print(f"Last adapter saved -> {last_dir}", flush=True)

    save_run_metadata(output_dir, args, train_df, valid_df, best_val_acc, best_val_loss)
    print(f"\nFinished. Best val_acc={best_val_acc:.4f}, best val_loss={best_val_loss:.4f}", flush=True)

    if test_csv:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_test_inference(
            test_csv=test_csv,
            image_root=image_root,
            output_dir=output_dir,
            args=args,
            device=device,
            use_4bit=use_4bit,
        )


if __name__ == "__main__":
    main()
