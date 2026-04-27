from __future__ import annotations
"""Core multistage VQA components used by the final solution.

High-level flow:
1. Brain planner decides question type and focus objects.
2. Grounding DINO and SAM optionally build localized crops and a focus panel.
3. One or more student VLM backends score answer choices.
4. Student probabilities are fused with lightweight priors for the final answer.
"""

import json
import math
import random
import re
import gc
import hashlib
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torch.utils.data import Dataset

from transformers import (
    AutoModelForImageTextToText,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    BitsAndBytesConfig,
    GroundingDinoProcessor,
    InternVLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    SamModel,
    SamProcessor,
)
try:
    from transformers import Qwen3_5ForConditionalGeneration
except Exception:  # pragma: no cover - optional at import time
    Qwen3_5ForConditionalGeneration = None

try:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except Exception:  # pragma: no cover - optional at import time
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def compute_adapter_cache_key(adapter_path: Path) -> str:
    stats = adapter_path.stat()
    raw = f"{adapter_path.resolve()}::{stats.st_size}::{stats.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def resolve_adapter_dir(adapter_dir: Optional[Path]) -> Optional[Path]:
    # Adapter ZIP files are extracted into a stable cache so downstream code can
    # treat both directories and ZIP archives the same way.
    if adapter_dir is None:
        return None

    resolved = adapter_dir.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Adapter path not found: {resolved}")
    if resolved.is_dir():
        return resolved
    if resolved.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported adapter path: {resolved}. Expected a directory or .zip archive.")

    cache_root = resolved.parent / ".adapter_cache"
    cache_dir = cache_root / f"{resolved.stem}_{compute_adapter_cache_key(resolved)}"
    marker_path = cache_dir / ".extract_complete"

    if marker_path.exists() and (cache_dir / "adapter_config.json").exists():
        return cache_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(resolved) as archive:
        archive.extractall(cache_dir)
    marker_path.write_text(f"source_zip={resolved}\n", encoding="utf-8")
    return cache_dir


def infer_base_model_from_adapter(adapter_dir: Optional[Path]) -> Optional[str]:
    # When an adapter declares its base model, prefer that model id at load time.
    if adapter_dir is None:
        return None
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = str(payload.get("base_model_name_or_path") or "").strip()
    return base_model or None


Image.MAX_IMAGE_PIXELS = None

CHOICE_LABELS = ["a", "b", "c", "d"]

SYSTEM_INSTRUCT = (
    "당신은 재활용품 이미지를 분석하는 전문가입니다. "
    "반드시 이미지에서 직접 보이는 정보만 사용하세요. "
    "질문의 대상, 재질, 종류, 상태와 상식적으로 맞지 않는 보기는 먼저 제외하세요. "
    "예를 들어 플라스틱을 묻는데 종이상자처럼 질문과 모순되거나, 플라스틱 캔처럼 비현실적인 조합은 정답 후보에서 우선 배제하세요. "
    "작은 글씨, 숫자, 라벨, 인쇄 문구, 재질 표기가 중요하면 해당 부분을 더 가까이 확대해 본다고 생각하고 다시 확인하세요. "
    "정답은 a, b, c, d 중 정확히 하나의 소문자 한 글자로만 답하세요."
)

BRAIN_SYSTEM_INSTRUCT = (
    "당신은 재활용품 VQA 문제를 위한 시각적 계획 수립 도우미입니다. "
    "질문 의도를 분류하고, 실제로 주목해야 할 객체 표현을 한국어 또는 짧은 영어 구로 뽑아야 합니다. "
    "반드시 JSON만 출력하세요."
)

GLOBAL_PROMPT_HINTS = [
    "질문의 핵심 대상과 보기의 의미가 서로 맞는지 먼저 확인하고, 질문과 상식에 어긋나는 보기는 빠르게 제외하세요.",
    "예: 플라스틱을 묻는데 종이상자, 금속/캔을 묻는데 비닐봉지, 실제로 어색한 조합인 플라스틱 캔 같은 보기는 우선 배제하세요.",
    "라벨, 숫자, 글자, 재질 표기, 작은 부품이 중요하면 그 부분만 다시 크게 본다고 생각하고 확인하세요.",
    "대상이 작거나 멀리 있으면 주변 배경보다 대상 중심으로 다시 좁혀 본다고 생각하고 판단하세요.",
]

MATERIAL_CRITERIA_HINT = (
    "재질 기준: 플라스틱=반투명/불투명, 매끄러운 표면, 가벼워 보임; "
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

QUESTION_TYPE_RULES = [
    (
        "count",
        [
            "몇 개",
            "몇개",
            "개수",
            "몇 개인가",
            "몇 개입니까",
            "몇 개 있",
            "수량",
            "몇 권",
            "몇 병",
            "총 몇",
        ],
    ),
    ("color", ["색깔", "색상", "무슨 색", "어떤 색", "뚜껑 색", "빨대의 색"]),
    ("location", ["어디에", "어느 쪽", "위치는", "놓여 있", "어디에 있"]),
    ("dominant", ["가장 많이", "가장 많은", "가장 많이 보이는", "주로 어떤", "주로 보이는"]),
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

OBJECT_LEXICON = [
    "플라스틱 병",
    "유리병",
    "유리 병",
    "플라스틱 컵",
    "컵",
    "병",
    "캔",
    "상자",
    "봉투",
    "포장지",
    "포장재",
    "용기",
    "빨대",
    "텀블러",
    "뚜껑",
    "컵홀더",
    "종이팩",
    "스티로폼",
    "분무기",
    "마커",
    "라벨",
]

OBJECT_PHRASE_ALIASES = {
    "플라스틱 포장재": ["포장재", "플라스틱 포장", "비닐", "plastic packaging", "plastic wrapper"],
    "플라스틱 병": ["병", "플라스틱병", "plastic bottle"],
    "유리병": ["유리 병", "병", "glass bottle"],
    "유리 병": ["유리병", "병", "glass bottle"],
    "금속 캔": ["캔", "음료 캔", "metal can"],
    "캔": ["금속 캔", "metal can"],
    "종이 봉투": ["봉투", "paper bag"],
    "봉투": ["종이 봉투", "paper bag"],
    "플라스틱 컵": ["컵", "plastic cup"],
    "컵": ["플라스틱 컵", "cup"],
    "종이팩": ["팩", "carton"],
    "스티로폼": ["foam", "styrofoam"],
    "뚜껑": ["캡", "cap", "lid"],
}

ROBUST_WASTE_PROMPT_BANK = {
    "soft_plastic": [
        "soft plastic",
        "thin plastic bag",
        "flexible plastic bag or wrap",
        "crumpled translucent plastic bag or wrap",
        "plastic wrapper",
        "plastic packaging",
    ],
    "rigid_plastic": [
        "rigid plastic",
        "hard plastic container",
        "solid rigid plastic container or bottle",
        "hollow rigid plastic container or bottle",
        "plastic bottle",
        "plastic cup",
    ],
    "metal": [
        "metal",
        "metal can",
        "shiny metallic can",
        "shiny reflective metal can or tin",
        "recyclable metal can",
    ],
    "cardboard": [
        "cardboard",
        "brown cardboard box",
        "stiff brown cardboard box",
        "thick brown cardboard box or packaging",
        "cardboard packaging",
    ],
    "paper": [
        "paper bag",
        "brown paper bag",
        "paper carton",
        "beverage carton",
        "paper cup holder",
    ],
    "glass": [
        "glass bottle",
        "transparent glass bottle",
        "clear glass bottle",
    ],
    "foam": [
        "white foam container",
        "styrofoam food container",
        "foam tray",
    ],
}

WASTE_CATEGORY_HINTS = {
    "soft_plastic": [
        "플라스틱 포장재",
        "비닐",
        "랩",
        "포장지",
        "포장재",
        "봉지",
        "wrapper",
        "soft plastic",
    ],
    "rigid_plastic": [
        "플라스틱 병",
        "플라스틱 컵",
        "플라스틱 용기",
        "용기",
        "텀블러",
        "분무기",
        "rigid plastic",
        "plastic bottle",
        "plastic container",
    ],
    "metal": [
        "금속 캔",
        "캔",
        "알루미늄",
        "철",
        "metal",
        "tin",
        "can",
    ],
    "cardboard": [
        "상자",
        "박스",
        "골판지",
        "cardboard",
        "box",
        "packaging",
    ],
    "paper": [
        "종이 봉투",
        "종이팩",
        "컵홀더",
        "paper bag",
        "paper carton",
        "carton",
        "cup holder",
    ],
    "glass": [
        "유리병",
        "유리 병",
        "glass bottle",
        "glass",
    ],
    "foam": [
        "스티로폼",
        "foam",
        "styrofoam",
    ],
}

CHOICE_GROUNDING_PROMPTS = {
    "플라스틱 포장재": [
        "crumpled translucent plastic bag or wrap",
        "flexible plastic bag or wrap",
        "thin plastic bag",
        "plastic wrapper",
    ],
    "플라스틱 병": [
        "solid rigid plastic container or bottle",
        "hollow rigid plastic container or bottle",
        "plastic bottle",
    ],
    "플라스틱 컵": [
        "clear plastic cup",
        "rigid plastic cup",
        "plastic cup",
    ],
    "금속 캔": [
        "shiny reflective metal can or tin",
        "shiny metallic can",
        "metal can",
    ],
    "종이 봉투": [
        "paper bag",
        "brown paper bag",
    ],
    "종이팩": [
        "paper carton",
        "beverage carton",
        "milk carton",
    ],
    "상자": [
        "thick brown cardboard box or packaging",
        "stiff brown cardboard box",
        "cardboard box",
    ],
    "유리병": [
        "transparent glass bottle",
        "clear glass bottle",
        "glass bottle",
    ],
    "유리 병": [
        "transparent glass bottle",
        "clear glass bottle",
        "glass bottle",
    ],
    "스티로폼": [
        "white foam container",
        "styrofoam food container",
        "foam tray",
    ],
    "컵홀더": [
        "paper cup holder",
        "cardboard cup holder",
    ],
}

GROUNDING_MATERIAL_WORDS = ["플라스틱", "유리", "금속", "종이", "비닐"]

# Student fusion weights are intentionally qtype-aware because the two backends
# are not equally strong on every question family.
QUESTION_TYPE_WEIGHTS = {
    "count": {"qwen": 0.45, "internvl": 0.55},
    "dominant": {"qwen": 0.45, "internvl": 0.55},
    "location": {"qwen": 0.45, "internvl": 0.55},
    "material": {"qwen": 0.55, "internvl": 0.45},
    "recycle": {"qwen": 0.60, "internvl": 0.40},
    "state": {"qwen": 0.55, "internvl": 0.45},
    "type": {"qwen": 0.55, "internvl": 0.45},
    "color": {"qwen": 0.50, "internvl": 0.50},
    "other": {"qwen": 0.50, "internvl": 0.50},
}


def has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"


def get_runtime_dtype(device: Optional[str] = None, prefer_float32: bool = False) -> torch.dtype:
    device = device or get_default_device()
    if prefer_float32:
        return torch.float32
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def supports_kbit_quantization(device: Optional[str] = None) -> bool:
    return (device or get_default_device()) == "cuda"


def should_use_device_map(device: Optional[str] = None) -> bool:
    return (device or get_default_device()) == "cuda"


def model_is_large_for_local(model_id: str) -> bool:
    match = re.search(r"[-_/](\d+)[bB](?:[-_/]|$)", model_id)
    return bool(match and int(match.group(1)) >= 16)


def resolve_model_id_for_local(
    model_id: Optional[str],
    fallback_model_id: Optional[str] = None,
    use_fallback: bool = False,
) -> Optional[str]:
    if not model_id:
        return None
    if not use_fallback or get_default_device() == "cuda" or not fallback_model_id:
        return model_id
    if model_is_large_for_local(model_id):
        return fallback_model_id
    return model_id


def build_model_load_kwargs(
    *,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    quant_4bit: bool = False,
    prefer_float32: bool = False,
) -> Dict[str, Any]:
    device = device or get_default_device()
    kwargs: Dict[str, Any] = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    if quant_4bit and supports_kbit_quantization(device):
        kwargs["quantization_config"] = get_4bit_config(dtype=get_runtime_dtype(device))
        kwargs["device_map"] = "auto"
        return kwargs

    kwargs["dtype"] = get_runtime_dtype(device, prefer_float32=prefer_float32)
    if should_use_device_map(device):
        kwargs["device_map"] = "auto"
    else:
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def finalize_loaded_model(model: torch.nn.Module, device: Optional[str] = None) -> torch.nn.Module:
    device = device or get_default_device()
    if not should_use_device_map(device):
        model = model.to(device)
    return model


def clear_runtime_memory(device: Optional[str] = None) -> None:
    device = device or get_default_device()
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and has_mps():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in str(text) for keyword in keywords)


def is_numeric_choice(text: str) -> bool:
    text = str(text).strip()
    return any(ch.isdigit() for ch in text) or text.endswith("개") or text.endswith("명")


def classify_question_type(question: str, choice_map: Dict[str, str]) -> str:
    question = str(question)
    for question_type, keywords in QUESTION_TYPE_RULES:
        if contains_any(question, keywords):
            return question_type
    numeric_choices = sum(is_numeric_choice(choice_map[label]) for label in CHOICE_LABELS)
    if numeric_choices >= 3:
        return "count"
    return "other"


def build_choices(row: pd.Series, shuffle: bool = False) -> Tuple[Dict[str, str], Optional[str]]:
    pairs = [(label, str(row[label])) for label in CHOICE_LABELS]
    if shuffle:
        pairs = random.sample(pairs, k=len(pairs))

    choice_map = {new_label: text for new_label, (_, text) in zip(CHOICE_LABELS, pairs)}

    remapped_answer = None
    answer = str(row["answer"]).strip().lower() if "answer" in row.index else None
    if answer in CHOICE_LABELS:
        for new_label, (orig_label, _) in zip(CHOICE_LABELS, pairs):
            if orig_label == answer:
                remapped_answer = new_label
                break

    return choice_map, remapped_answer


def normalize_match_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text).strip().lower())


def focus_matches_choice_text(focus_text: str, choice_text: str) -> bool:
    focus_norm = normalize_match_text(focus_text)
    choice_norm = normalize_match_text(choice_text)
    if not focus_norm or not choice_norm:
        return False
    return focus_norm in choice_norm or choice_norm in focus_norm


def extract_choice(text: str) -> str:
    text = str(text).strip().lower()
    if not text:
        return "a"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for candidate in reversed(lines or [text]):
        if candidate in CHOICE_LABELS:
            return candidate
        for token in re.findall(r"[a-d]", candidate):
            if token in CHOICE_LABELS:
                return token
    return "a"


def apply_chat_template_safe(processor: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def get_model_device(model: torch.nn.Module) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return get_default_device()


def get_4bit_config(dtype: torch.dtype = torch.bfloat16) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = candidate.replace("\n", " ")
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


def normalize_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return x1, y1, x2, y2


def box_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / max(1, union)


def box_area(box: Sequence[int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def box_area_ratio(box: Sequence[int], width: int, height: int) -> float:
    return box_area(box) / max(1, width * height)


def box_containment_ratio(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    return inter_area / max(1, box_area(box_a))


def nms_candidates(candidates: Sequence["BoxCandidate"], iou_threshold: float = 0.45) -> List["BoxCandidate"]:
    kept: List[BoxCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(box_iou(candidate.box, existing.box) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def mask_bbox(mask: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = torch.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1
    return x1, y1, x2, y2


def heuristic_focus_objects(question: str, choice_map: Dict[str, str]) -> List[str]:
    hits = [obj for obj in OBJECT_LEXICON if obj in question]
    if hits:
        return hits[:3]

    options_blob = " ".join(str(choice_map.get(label, "")) for label in CHOICE_LABELS)
    hits = [obj for obj in OBJECT_LEXICON if obj in options_blob]
    if hits:
        return hits[:3]
    return ["재활용품"]


def unique_phrases(phrases: Sequence[str]) -> List[str]:
    seen = set()
    results = []
    for phrase in phrases:
        cleaned = str(phrase).strip()
        if not cleaned:
            continue
        key = normalize_match_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results


def matches_any_alias(text: str, aliases: Sequence[str]) -> bool:
    text_norm = normalize_match_text(text)
    if not text_norm:
        return False
    for alias in aliases:
        alias_norm = normalize_match_text(alias)
        if not alias_norm:
            continue
        if alias_norm in text_norm or text_norm in alias_norm:
            return True
    return False


def simplify_grounding_phrase(phrase: str) -> List[str]:
    variants = [phrase]
    stripped = phrase
    for material_word in GROUNDING_MATERIAL_WORDS:
        stripped = stripped.replace(material_word, "").strip()
    if stripped and stripped != phrase:
        variants.append(stripped)
    if " " in phrase:
        variants.append(phrase.split()[-1].strip())
    return unique_phrases(variants)


def match_waste_categories(text: str) -> List[str]:
    matched = [
        category
        for category, aliases in WASTE_CATEGORY_HINTS.items()
        if matches_any_alias(text, aliases)
    ]
    return list(dict.fromkeys(matched))


def build_choice_grounding_prompts(text: str) -> List[str]:
    prompts: List[str] = []
    for key, aliases in CHOICE_GROUNDING_PROMPTS.items():
        if focus_matches_choice_text(text, key):
            prompts.extend(aliases)
    if not prompts:
        for category in match_waste_categories(text):
            prompts.extend(ROBUST_WASTE_PROMPT_BANK.get(category, []))

    prompts.extend(alias for alias in simplify_grounding_phrase(text) if re.search(r"[a-zA-Z]", alias))
    return unique_phrases(prompts)


def expand_grounding_phrases(
    focus_objects: Sequence[str],
    question_type: str,
) -> List[str]:
    phrases: List[str] = []
    for focus in focus_objects:
        phrases.extend(simplify_grounding_phrase(focus))
        for key, aliases in OBJECT_PHRASE_ALIASES.items():
            if focus_matches_choice_text(focus, key):
                phrases.append(key)
                phrases.extend(aliases)
    if question_type == "count":
        phrases.extend(
            alias
            for focus in focus_objects
            for key, aliases in OBJECT_PHRASE_ALIASES.items()
            if focus_matches_choice_text(focus, key)
            for alias in aliases[:2]
        )
    return unique_phrases(phrases)


def unique_grounding_queries(queries: Sequence["GroundingQuery"]) -> List["GroundingQuery"]:
    seen = set()
    results: List[GroundingQuery] = []
    for query in queries:
        phrase = str(query.phrase).strip()
        if not phrase:
            continue
        key = (
            normalize_match_text(phrase),
            str(query.choice_label).strip().lower(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        results.append(GroundingQuery(phrase=phrase, choice_label=query.choice_label, source=query.source))
    return results


def build_grounding_queries(
    focus_objects: Sequence[str],
    question_type: str,
    choice_map: Optional[Dict[str, str]],
    config: "MultiStageConfig",
) -> List[GroundingQuery]:
    queries: List[GroundingQuery] = [
        GroundingQuery(phrase=phrase, source="focus")
        for phrase in expand_grounding_phrases(focus_objects, question_type)
    ]

    if config.grounding_prompt_strategy == "robust_waste":
        for focus in focus_objects:
            for phrase in build_choice_grounding_prompts(focus):
                queries.append(GroundingQuery(phrase=phrase, source="focus_optimized"))

    if (
        config.grounding_use_choice_prompts
        and choice_map
        and question_type in config.grounding_choice_prompt_question_types
    ):
        for label in CHOICE_LABELS:
            choice_text = str(choice_map.get(label, "")).strip()
            if not choice_text:
                continue
            prompts = unique_phrases(
                simplify_grounding_phrase(choice_text)
                + build_choice_grounding_prompts(choice_text)
            )[: config.grounding_choice_prompt_limit]
            for phrase in prompts:
                queries.append(GroundingQuery(phrase=phrase, choice_label=label, source="choice"))

    return unique_grounding_queries(queries)[: config.grounding_query_limit]


def parse_numeric_choice_value(text: str) -> Optional[int]:
    match = re.search(r"\d+", str(text))
    if match:
        return int(match.group(0))
    return None


def extract_numeric_choice_values(choice_map: Dict[str, str]) -> List[int]:
    values: List[int] = []
    for label in CHOICE_LABELS:
        value = parse_numeric_choice_value(choice_map.get(label, ""))
        if value is not None:
            values.append(value)
    return values


def default_brain_plan(question: str, choice_map: Dict[str, str]) -> "BrainPlan":
    question_type = classify_question_type(question, choice_map)
    focus_objects = heuristic_focus_objects(question, choice_map)
    answer_strategy = {
        "count": "질문에서 지정한 대상만 한 번씩 세기",
        "material": "재질 단서와 광택/투명도 보기",
        "recycle": "재질과 용도에 맞는 분리배출 분류 선택",
        "color": "질문 대상의 실제 주된 색상 비교",
        "location": "질문 대상을 먼저 찾고 상대 위치 판단",
        "dominant": "각 보기 후보를 대략 세어 가장 많은 것 선택",
        "state": "대상의 세부 상태나 속성에 집중",
        "type": "대상을 특정한 뒤 보기 차이를 비교",
    }.get(question_type, "이미지 전체와 보기 차이를 비교")
    return BrainPlan(
        question_type=question_type,
        focus_objects=focus_objects,
        answer_strategy=answer_strategy,
        raw_text="heuristic",
    )


def build_mc_prompt(question: str, choice_map: Dict[str, str], extra_context: str = "") -> str:
    question = str(question).strip()
    lines = [f"질문: {question}"]

    if GLOBAL_PROMPT_HINTS:
        lines.extend(["", "공통 원칙:"])
        lines.extend(f"- {hint}" for hint in GLOBAL_PROMPT_HINTS if hint)

    question_type = classify_question_type(question, choice_map)
    prompt_hints = PROMPT_HINTS.get(question_type, PROMPT_HINTS["other"])
    if prompt_hints:
        lines.extend(["", "유형별 판단 기준:"])
        lines.extend(f"- {hint}" for hint in prompt_hints if hint)

    if extra_context.strip():
        lines.extend(["", "[참고 맥락]", extra_context.strip()])

    lines.append("")
    for label in CHOICE_LABELS:
        lines.append(f"({label}) {choice_map[label]}")
    lines.extend(["", "정답:"])
    return "\n".join(lines)


@dataclass
class BrainPlan:
    question_type: str
    focus_objects: List[str] = field(default_factory=list)
    visual_attributes: List[str] = field(default_factory=list)
    answer_strategy: str = ""
    confidence: float = 0.0
    raw_text: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "question_type": self.question_type,
            "focus_objects": self.focus_objects,
            "visual_attributes": self.visual_attributes,
            "answer_strategy": self.answer_strategy,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
        }


def brain_plan_from_json_dict(data: Dict[str, Any]) -> BrainPlan:
    return BrainPlan(
        question_type=str(data.get("question_type") or "other"),
        focus_objects=[str(item) for item in data.get("focus_objects", []) if str(item).strip()],
        visual_attributes=[str(item) for item in data.get("visual_attributes", []) if str(item).strip()],
        answer_strategy=str(data.get("answer_strategy") or ""),
        confidence=float(data.get("confidence") or 0.0),
        raw_text=str(data.get("raw_text") or ""),
    )


@dataclass
class BoxCandidate:
    phrase: str
    score: float
    box: Tuple[int, int, int, int]
    label: str = ""
    choice_label: str = ""
    source: str = ""


@dataclass
class CropCandidate:
    phrase: str
    score: float
    box: Tuple[int, int, int, int]
    crop_box: Tuple[int, int, int, int]
    mask_score: float = 0.0


@dataclass
class GroundingQuery:
    phrase: str
    choice_label: str = ""
    source: str = "focus"


@dataclass
class PreparedContext:
    sample_id: str
    question_type: str
    focus_objects: List[str]
    answer_strategy: str
    detections: List[BoxCandidate]
    crops: List[CropCandidate]
    brain_confidence: float = 0.0
    grounding_quality: float = 0.0
    grounding_skip_reason: str = ""
    focus_panel_path: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question_type": self.question_type,
            "focus_objects": self.focus_objects,
            "answer_strategy": self.answer_strategy,
            "detections": [asdict(item) for item in self.detections],
            "crops": [asdict(item) for item in self.crops],
            "brain_confidence": self.brain_confidence,
            "grounding_quality": self.grounding_quality,
            "grounding_skip_reason": self.grounding_skip_reason,
            "focus_panel_path": self.focus_panel_path,
        }


@dataclass
class ModelPrediction:
    model_name: str
    answer: str
    confidence: float
    choice_probs: Dict[str, float]
    raw_scores: Dict[str, float]


@dataclass
class EnsemblePrediction:
    answer: str
    confidence: float
    aggregated_probs: Dict[str, float]
    model_predictions: Dict[str, ModelPrediction]
    context: PreparedContext


@dataclass
class StudentModelConfig:
    # Per-student settings are kept separate so Qwen and InternVL can be enabled
    # independently and use different adapters if needed.
    name: str
    model_id: str
    adapter_path: Optional[str] = None
    enabled: bool = True
    quant_4bit: bool = True
    weight: float = 1.0


@dataclass
class MultiStageConfig:
    # This dataclass centralizes every knob that affects the multistage runtime:
    # brain planning, grounding, priors, cache paths, and student backends.
    seed: int = 42
    min_pixels: int = 256 * 256
    max_pixels: int = 512 * 512
    brain_model_id: Optional[str] = "Qwen/Qwen3-VL-32B-Instruct"
    local_brain_fallback_model_id: Optional[str] = "Qwen/Qwen3-VL-4B-Instruct"
    prefer_local_brain_fallback_on_non_cuda: bool = True
    grounding_model_id: Optional[str] = "IDEA-Research/grounding-dino-base"
    sam_model_id: Optional[str] = "facebook/sam-vit-base"
    brain_quant_4bit: bool = True
    grounding_prompt_strategy: str = "robust_waste"
    grounding_use_choice_prompts: bool = True
    grounding_choice_prompt_question_types: Tuple[str, ...] = ("recycle", "material", "type", "dominant")
    grounding_choice_prompt_limit: int = 4
    grounding_query_limit: int = 18
    skip_grounding_when_focus_matches_choice: bool = True
    brain_skip_grounding_confidence: float = 0.90
    skip_grounding_question_types: Tuple[str, ...] = ("recycle", "type")
    grounding_threshold: float = 0.25
    grounding_text_threshold: float = 0.20
    grounding_iou_threshold: float = 0.45
    grounding_threshold_count: float = 0.30
    grounding_iou_threshold_count: float = 0.35
    grounding_max_box_area_ratio: float = 0.80
    grounding_max_box_area_ratio_count: float = 0.45
    grounding_min_box_area_ratio: float = 0.001
    grounding_containment_threshold: float = 0.90
    brain_choice_prior_weight: float = 0.18
    detection_choice_prior_weight: float = 0.14
    detection_choice_min_top_score: float = 0.24
    detection_choice_min_margin: float = 0.03
    detection_count_prior_weight: float = 0.16
    detection_count_prior_max_numeric_choice: int = 6
    detection_quality_threshold_for_prior: float = 0.45
    max_focus_objects: int = 3
    max_boxes: int = 4
    panel_tile_size: int = 320
    brain_max_new_tokens: int = 256
    brain_cache_dir: str = "artifacts/brain_cache"
    panel_dir: str = "artifacts/focus_panels"
    cache_dir: str = "artifacts/context_cache"
    qwen_student: StudentModelConfig = field(
        default_factory=lambda: StudentModelConfig(
            name="qwen",
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            enabled=True,
            quant_4bit=True,
            weight=1.0,
        )
    )
    internvl_student: StudentModelConfig = field(
        default_factory=lambda: StudentModelConfig(
            name="internvl",
            model_id="OpenGVLab/InternVL3-1B-hf",
            enabled=False,
            quant_4bit=True,
            weight=1.0,
        )
    )


class BrainPlanner:
    def __init__(self, config: MultiStageConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.device = get_default_device()

    def is_enabled(self) -> bool:
        return bool(self.config.brain_model_id)

    def load(self) -> None:
        if not self.is_enabled() or self.model is not None:
            return
        brain_model_id = resolve_model_id_for_local(
            self.config.brain_model_id,
            fallback_model_id=self.config.local_brain_fallback_model_id,
            use_fallback=self.config.prefer_local_brain_fallback_on_non_cuda,
        )
        if not brain_model_id:
            return

        processor = AutoProcessor.from_pretrained(
            brain_model_id,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
            trust_remote_code=True,
        )

        kwargs = build_model_load_kwargs(
            device=self.device,
            trust_remote_code=True,
            quant_4bit=self.config.brain_quant_4bit,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            brain_model_id,
            **kwargs,
        )

        self.processor = processor
        self.model = finalize_loaded_model(model, self.device).eval()
        self.device = get_model_device(self.model)

    def build_prompt(self, question: str, choice_map: Dict[str, str]) -> str:
        return (
            "다음 객관식 시각질문 문제를 분석해 주세요.\n"
            "JSON 스키마는 다음과 같습니다.\n"
            "{\n"
            '  "question_type": "count|color|location|dominant|recycle|material|state|type|other",\n'
            '  "focus_objects": ["주목할 객체 표현", "..."],\n'
            '  "visual_attributes": ["색, 재질, 상태 등 단서", "..."],\n'
            '  "answer_strategy": "짧은 풀이 전략",\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            "규칙:\n"
            "- focus_objects는 Grounding DINO에 넣을 짧은 구로 작성하세요.\n"
            "- 너무 넓은 표현보다 실제로 찾을 수 있는 대상명 위주로 적으세요.\n"
            "- JSON 외 다른 문장은 출력하지 마세요.\n\n"
            f"질문: {question}\n"
            f"(a) {choice_map['a']}\n"
            f"(b) {choice_map['b']}\n"
            f"(c) {choice_map['c']}\n"
            f"(d) {choice_map['d']}\n"
        )

    def plan(self, image: Image.Image, question: str, choice_map: Dict[str, str]) -> BrainPlan:
        fallback = default_brain_plan(question, choice_map)
        if not self.is_enabled():
            return fallback

        try:
            self.load()
            messages = [
                {"role": "system", "content": [{"type": "text", "text": BRAIN_SYSTEM_INSTRUCT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.build_prompt(question, choice_map)},
                    ],
                },
            ]
            text = apply_chat_template_safe(self.processor, messages, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            inputs = move_batch_to_device(inputs, self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.brain_max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            output_text = self.processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
            parsed = safe_json_loads(output_text)
            if not parsed:
                fallback.raw_text = output_text
                return fallback

            question_type = str(parsed.get("question_type") or fallback.question_type).strip().lower()
            focus_objects = [
                str(item).strip()
                for item in parsed.get("focus_objects", [])
                if str(item).strip()
            ][: self.config.max_focus_objects]
            if not focus_objects:
                focus_objects = fallback.focus_objects

            return BrainPlan(
                question_type=question_type if question_type else fallback.question_type,
                focus_objects=focus_objects,
                visual_attributes=[
                    str(item).strip()
                    for item in parsed.get("visual_attributes", [])
                    if str(item).strip()
                ],
                answer_strategy=str(parsed.get("answer_strategy") or fallback.answer_strategy).strip(),
                confidence=float(parsed.get("confidence") or 0.0),
                raw_text=output_text,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            fallback.raw_text = f"planner_error: {exc}"
            return fallback


def grounding_skip_reason(
    plan: BrainPlan,
    choice_map: Dict[str, str],
    config: MultiStageConfig,
) -> Optional[str]:
    if not config.skip_grounding_when_focus_matches_choice:
        return None
    if plan.question_type not in config.skip_grounding_question_types:
        return None
    if plan.confidence < config.brain_skip_grounding_confidence:
        return None

    matched_focus_objects = [
        focus
        for focus in plan.focus_objects
        if any(focus_matches_choice_text(focus, choice_map[label]) for label in CHOICE_LABELS)
    ]
    if not matched_focus_objects:
        return None

    return (
        "brain focus matched choice text with high confidence: "
        + ", ".join(matched_focus_objects)
    )


def filter_grounding_candidates(
    candidates: Sequence[BoxCandidate],
    width: int,
    height: int,
    question_type: str,
    config: MultiStageConfig,
) -> List[BoxCandidate]:
    max_area_ratio = (
        config.grounding_max_box_area_ratio_count
        if question_type == "count"
        else config.grounding_max_box_area_ratio
    )
    min_score = (
        config.grounding_threshold_count
        if question_type == "count"
        else config.grounding_threshold
    )
    iou_threshold = (
        config.grounding_iou_threshold_count
        if question_type == "count"
        else config.grounding_iou_threshold
    )

    filtered: List[BoxCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        area_ratio = box_area_ratio(candidate.box, width, height)
        if candidate.score < min_score:
            continue
        if area_ratio < config.grounding_min_box_area_ratio or area_ratio > max_area_ratio:
            continue

        redundant = False
        for existing in filtered:
            if box_containment_ratio(candidate.box, existing.box) >= config.grounding_containment_threshold:
                redundant = True
                break
            if (
                box_containment_ratio(existing.box, candidate.box) >= config.grounding_containment_threshold
                and box_area(candidate.box) > box_area(existing.box)
            ):
                redundant = True
                break
        if not redundant:
            filtered.append(candidate)

    return nms_candidates(filtered, iou_threshold=iou_threshold)[: config.max_boxes]


def grounding_quality_score(
    candidates: Sequence[BoxCandidate],
    width: int,
    height: int,
    question_type: str,
    config: MultiStageConfig,
) -> float:
    if not candidates:
        return 0.0

    area_ratios = [box_area_ratio(candidate.box, width, height) for candidate in candidates]
    scores = [candidate.score for candidate in candidates]
    max_area_ratio = (
        config.grounding_max_box_area_ratio_count
        if question_type == "count"
        else config.grounding_max_box_area_ratio
    )
    good_area_fraction = sum(
        config.grounding_min_box_area_ratio <= ratio <= max_area_ratio
        for ratio in area_ratios
    ) / len(area_ratios)
    quality = (sum(scores) / len(scores)) * (0.55 + 0.45 * good_area_fraction)
    if question_type == "count" and max(area_ratios, default=0.0) > max_area_ratio:
        quality *= 0.7
    return float(max(0.0, min(1.0, quality)))


def build_brain_choice_prior(
    context: PreparedContext,
    choice_map: Dict[str, str],
    config: MultiStageConfig,
) -> Tuple[Optional[Dict[str, float]], float]:
    if context.question_type not in config.skip_grounding_question_types:
        return None, 0.0
    if context.brain_confidence < config.brain_skip_grounding_confidence:
        return None, 0.0

    matched_labels = [
        label
        for label in CHOICE_LABELS
        if any(focus_matches_choice_text(focus, choice_map[label]) for focus in context.focus_objects)
    ]
    matched_labels = list(dict.fromkeys(matched_labels))
    if len(matched_labels) != 1:
        return None, 0.0

    prior = {label: 0.0 for label in CHOICE_LABELS}
    prior[matched_labels[0]] = 1.0
    return prior, config.brain_choice_prior_weight


def build_detection_choice_prior(
    context: PreparedContext,
    choice_map: Dict[str, str],
    config: MultiStageConfig,
) -> Tuple[Optional[Dict[str, float]], float]:
    if context.question_type not in config.grounding_choice_prompt_question_types:
        return None, 0.0
    if context.grounding_quality < config.detection_quality_threshold_for_prior:
        return None, 0.0

    scores = {label: 0.0 for label in CHOICE_LABELS}
    for candidate in context.detections:
        if candidate.choice_label in scores:
            scores[candidate.choice_label] += max(0.0, float(candidate.score))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < config.detection_choice_min_top_score:
        return None, 0.0
    if top_score - second_score < config.detection_choice_min_margin:
        return None, 0.0

    prior = normalize_choice_probs(scores)
    weight = config.detection_choice_prior_weight * context.grounding_quality
    if choice_map.get(top_label):
        return prior, weight
    return None, 0.0


def build_detection_count_prior(
    context: PreparedContext,
    choice_map: Dict[str, str],
    config: MultiStageConfig,
) -> Tuple[Optional[Dict[str, float]], float]:
    if context.question_type != "count":
        return None, 0.0
    if context.grounding_quality < config.detection_quality_threshold_for_prior:
        return None, 0.0

    numeric_choices = extract_numeric_choice_values(choice_map)
    if numeric_choices and max(numeric_choices) > config.detection_count_prior_max_numeric_choice:
        return None, 0.0

    detected_count = len(context.crops or context.detections)
    if detected_count <= 0:
        return None, 0.0

    matched_labels = [
        label
        for label in CHOICE_LABELS
        if parse_numeric_choice_value(choice_map[label]) == detected_count
    ]
    if len(matched_labels) != 1:
        return None, 0.0

    prior = {label: 0.0 for label in CHOICE_LABELS}
    prior[matched_labels[0]] = 1.0
    weight = config.detection_count_prior_weight * context.grounding_quality
    return prior, weight


def normalize_choice_probs(choice_probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(choice_probs.values()) or 1.0
    return {label: value / total for label, value in choice_probs.items()}


class GroundingDinoLocalizer:
    def __init__(self, config: MultiStageConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.device = get_default_device()

    def is_enabled(self) -> bool:
        return bool(self.config.grounding_model_id)

    def load(self) -> None:
        if not self.is_enabled() or self.model is not None:
            return
        self.processor = GroundingDinoProcessor.from_pretrained(self.config.grounding_model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.grounding_model_id,
            **build_model_load_kwargs(device=self.device, prefer_float32=True),
        )
        self.model = finalize_loaded_model(model, self.device).eval()
        self.device = get_model_device(self.model)

    def localize(
        self,
        image: Image.Image,
        focus_objects: Sequence[str],
        question_type: str = "other",
        choice_map: Optional[Dict[str, str]] = None,
    ) -> List[BoxCandidate]:
        if not self.is_enabled() or not focus_objects:
            return []

        self.load()
        width, height = image.size
        candidates: List[BoxCandidate] = []
        postprocess_threshold = (
            self.config.grounding_threshold_count
            if question_type == "count"
            else self.config.grounding_threshold
        )

        queries = build_grounding_queries(
            focus_objects=focus_objects,
            question_type=question_type,
            choice_map=choice_map,
            config=self.config,
        )
        for query in queries:
            prompt_text = query.phrase if query.phrase.endswith(".") else f"{query.phrase}."
            try:
                inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")
                inputs = move_batch_to_device(inputs, self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                result = self.processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs.get("input_ids"),
                    threshold=postprocess_threshold,
                    text_threshold=self.config.grounding_text_threshold,
                    target_sizes=[(height, width)],
                )[0]
                for score, box, label in zip(
                    result["scores"],
                    result["boxes"],
                    result.get("text_labels", result.get("labels", [])),
                ):
                    normalized = normalize_box(box.tolist(), width, height)
                    candidates.append(
                        BoxCandidate(
                            phrase=query.phrase,
                            score=float(score.item()),
                            box=normalized,
                            label=str(label),
                            choice_label=query.choice_label,
                            source=query.source,
                        )
                    )
            except Exception:
                continue

        return filter_grounding_candidates(candidates, width, height, question_type, self.config)


class SamRefiner:
    def __init__(self, config: MultiStageConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.device = get_default_device()

    def is_enabled(self) -> bool:
        return bool(self.config.sam_model_id)

    def load(self) -> None:
        if not self.is_enabled() or self.model is not None:
            return
        self.processor = SamProcessor.from_pretrained(self.config.sam_model_id)
        model = SamModel.from_pretrained(
            self.config.sam_model_id,
            **build_model_load_kwargs(device=self.device, prefer_float32=True),
        )
        self.model = finalize_loaded_model(model, self.device).eval()
        self.device = get_model_device(self.model)

    def refine(self, image: Image.Image, candidates: Sequence[BoxCandidate]) -> List[CropCandidate]:
        if not candidates:
            return []
        if not self.is_enabled():
            return [
                CropCandidate(
                    phrase=item.phrase,
                    score=item.score,
                    box=item.box,
                    crop_box=item.box,
                    mask_score=0.0,
                )
                for item in candidates
            ]

        self.load()
        refined: List[CropCandidate] = []

        for candidate in candidates:
            try:
                inputs = self.processor(
                    images=image,
                    input_boxes=[[[float(v) for v in candidate.box]]],
                    return_tensors="pt",
                )
                inputs = move_batch_to_device(inputs, self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, multimask_output=False)
                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.detach().cpu(),
                    inputs["original_sizes"].detach().cpu(),
                    inputs["reshaped_input_sizes"].detach().cpu(),
                    mask_threshold=0.0,
                )
                mask = masks[0]
                while mask.ndim > 2:
                    mask = mask[0]
                mask_box = mask_bbox(mask)
                crop_box = mask_box if mask_box is not None else candidate.box
                refined.append(
                    CropCandidate(
                        phrase=candidate.phrase,
                        score=candidate.score,
                        box=candidate.box,
                        crop_box=crop_box,
                        mask_score=float(outputs.iou_scores.max().item()),
                    )
                )
            except Exception:
                refined.append(
                    CropCandidate(
                        phrase=candidate.phrase,
                        score=candidate.score,
                        box=candidate.box,
                        crop_box=candidate.box,
                        mask_score=0.0,
                    )
                )

        return refined


def crop_with_margin(image: Image.Image, crop_box: Sequence[int], margin_ratio: float = 0.06) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = crop_box
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * margin_ratio)
    pad_y = int(h * margin_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return image.crop((x1, y1, x2, y2))


def compose_focus_panel(
    image: Image.Image,
    crops: Sequence[CropCandidate],
    tile_size: int = 320,
    max_items: int = 4,
) -> Optional[Image.Image]:
    usable = list(crops)[:max_items]
    if not usable:
        return None

    tiles: List[Image.Image] = []
    for index, crop in enumerate(usable, start=1):
        tile = Image.new("RGB", (tile_size, tile_size + 28), color="white")
        crop_image = crop_with_margin(image, crop.crop_box)
        crop_image = ImageOps.contain(crop_image, (tile_size, tile_size))
        offset_x = (tile_size - crop_image.width) // 2
        offset_y = (tile_size - crop_image.height) // 2
        tile.paste(crop_image, (offset_x, offset_y))
        draw = ImageDraw.Draw(tile)
        label = f"{index}. {crop.phrase[:22]}"
        draw.text((8, tile_size + 6), label, fill="black")
        tiles.append(tile)

    cols = min(2, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    panel = Image.new("RGB", (cols * tile_size, rows * (tile_size + 28)), color="#f7f7f7")
    for idx, tile in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        panel.paste(tile, (col * tile_size, row * (tile_size + 28)))
    return panel


def compose_student_image(image: Image.Image, focus_panel: Optional[Image.Image] = None) -> Image.Image:
    if focus_panel is None:
        return image

    target_width = max(image.width, focus_panel.width)
    resized_image = ImageOps.contain(image, (target_width, max(1, image.height)))
    resized_panel = ImageOps.contain(focus_panel, (target_width, max(1, focus_panel.height)))

    canvas = Image.new(
        "RGB",
        (target_width, resized_image.height + resized_panel.height + 60),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Original image", fill="black")
    canvas.paste(resized_image, (0, 28))
    draw.text((10, 28 + resized_image.height + 6), "Focus regions", fill="black")
    canvas.paste(resized_panel, (0, 28 + resized_image.height + 28))
    return canvas


def apply_student_sharpen(image: Image.Image, question_type: str) -> Image.Image:
    if str(question_type) == "count":
        return image
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))


def build_context_block(context: PreparedContext) -> str:
    lines = [
        f"질문 유형: {context.question_type}",
    ]
    if context.focus_objects:
        lines.append("주목 대상: " + ", ".join(context.focus_objects))
    if context.answer_strategy:
        lines.append("풀이 전략: " + context.answer_strategy)
    choice_hints = [item for item in context.detections if item.choice_label]
    if choice_hints:
        hint_text = "; ".join(
            f"{item.choice_label}:{item.phrase} (score={item.score:.2f})"
            for item in choice_hints[:4]
        )
        lines.append("검출 힌트: " + hint_text)
    if context.crops:
        crop_text = "; ".join(
            f"{item.phrase} (box_score={item.score:.2f}, mask_score={item.mask_score:.2f})"
            for item in context.crops
        )
        lines.append("확대 후보: " + crop_text)
        lines.append("아래 이미지에는 원본과 관심 영역 확대본이 함께 들어 있습니다.")
    return "\n".join(lines)


class StudentVLMBackend:
    def __init__(self, model_config: StudentModelConfig, min_pixels: int, max_pixels: int):
        self.model_config = model_config
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = None
        self.model = None
        self.device = get_default_device()
        self.resolved_adapter_path: Optional[str] = None
        self.effective_model_id = model_config.model_id

    def is_enabled(self) -> bool:
        return self.model_config.enabled

    def _load_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def load(self) -> None:
        if not self.is_enabled() or self.model is not None:
            return
        # Adapter resolution is handled here once so both Qwen and InternVL
        # backends can share the same base loading logic.
        resolved_adapter = None
        if self.model_config.adapter_path:
            resolved_adapter = resolve_adapter_dir(Path(self.model_config.adapter_path))
        adapter_base_model = infer_base_model_from_adapter(resolved_adapter)
        self.resolved_adapter_path = str(resolved_adapter) if resolved_adapter is not None else None
        self.effective_model_id = adapter_base_model or self.model_config.model_id
        self.processor = AutoProcessor.from_pretrained(
            self.effective_model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )
        model = self._load_model()
        self.model = finalize_loaded_model(model, self.device).eval()
        self.device = get_model_device(self.model)

    def _apply_adapter_if_needed(self, base_model: torch.nn.Module) -> torch.nn.Module:
        # If no adapter path is provided, the backend simply runs the base model.
        adapter_path = self.resolved_adapter_path or self.model_config.adapter_path
        if adapter_path and PeftModel is not None:
            return PeftModel.from_pretrained(base_model, adapter_path)
        return base_model

    def build_messages(
        self,
        image: Image.Image,
        question: str,
        choice_map: Dict[str, str],
        context: PreparedContext,
    ) -> List[Dict[str, Any]]:
        prompt = build_mc_prompt(
            question,
            choice_map,
            extra_context=build_context_block(context),
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def score_choices(
        self,
        image: Image.Image,
        question: str,
        choice_map: Dict[str, str],
        context: PreparedContext,
    ) -> ModelPrediction:
        self.load()
        messages = self.build_messages(image, question, choice_map, context)
        prompt_text = apply_chat_template_safe(self.processor, messages, add_generation_prompt=True)
        prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        prompt_inputs = move_batch_to_device(prompt_inputs, self.device)
        base_len = prompt_inputs["input_ids"].shape[1]

        raw_scores: Dict[str, float] = {}
        for label in CHOICE_LABELS:
            candidate_messages = messages + [
                {"role": "assistant", "content": [{"type": "text", "text": label}]}
            ]
            candidate_text = apply_chat_template_safe(
                self.processor,
                candidate_messages,
                add_generation_prompt=False,
            )
            candidate_inputs = self.processor(text=[candidate_text], images=[image], return_tensors="pt")
            candidate_inputs = move_batch_to_device(candidate_inputs, self.device)
            labels = candidate_inputs["input_ids"].clone()
            labels[:, :base_len] = -100
            with torch.no_grad():
                outputs = self.model(**candidate_inputs, labels=labels)
            target_count = int((labels != -100).sum().item())
            raw_scores[label] = -float(outputs.loss.item()) * max(target_count, 1)

        score_tensor = torch.tensor([raw_scores[label] for label in CHOICE_LABELS], dtype=torch.float32)
        probs_tensor = torch.softmax(score_tensor, dim=0)
        choice_probs = {
            label: float(probs_tensor[idx].item())
            for idx, label in enumerate(CHOICE_LABELS)
        }
        answer = max(choice_probs.items(), key=lambda item: item[1])[0]
        confidence = choice_probs[answer]

        return ModelPrediction(
            model_name=self.model_config.name,
            answer=answer,
            confidence=confidence,
            choice_probs=choice_probs,
            raw_scores=raw_scores,
        )


class QwenStudentBackend(StudentVLMBackend):
    def _load_model(self) -> torch.nn.Module:
        kwargs = build_model_load_kwargs(
            device=self.device,
            trust_remote_code=True,
            quant_4bit=self.model_config.quant_4bit,
        )
        model_id = self.effective_model_id
        if "qwen3.5" in str(model_id).lower():
            if Qwen3_5ForConditionalGeneration is not None:
                base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_id,
                    **kwargs,
                )
            else:
                base_model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    **kwargs,
                )
        else:
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                **kwargs,
            )
        return self._apply_adapter_if_needed(base_model)


class InternVLStudentBackend(StudentVLMBackend):
    def _load_model(self) -> torch.nn.Module:
        kwargs = build_model_load_kwargs(
            device=self.device,
            trust_remote_code=True,
            quant_4bit=self.model_config.quant_4bit,
        )
        base_model = InternVLForConditionalGeneration.from_pretrained(
            self.model_config.model_id,
            **kwargs,
        )
        return self._apply_adapter_if_needed(base_model)


class MultiStageVQAPipeline:
    def __init__(self, config: Optional[MultiStageConfig] = None):
        self.config = config or MultiStageConfig()
        set_seed(self.config.seed)

        self.panel_dir = Path(self.config.panel_dir)
        self.panel_dir.mkdir(parents=True, exist_ok=True)
        self.brain_cache_dir = Path(self.config.brain_cache_dir)
        self.brain_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.brain = BrainPlanner(self.config)
        self.localizer = GroundingDinoLocalizer(self.config)
        self.refiner = SamRefiner(self.config)
        self.backends: Dict[str, StudentVLMBackend] = {}
        if self.config.qwen_student.enabled:
            self.backends["qwen"] = QwenStudentBackend(
                self.config.qwen_student,
                self.config.min_pixels,
                self.config.max_pixels,
            )
        if self.config.internvl_student.enabled:
            self.backends["internvl"] = InternVLStudentBackend(
                self.config.internvl_student,
                self.config.min_pixels,
                self.config.max_pixels,
            )

    def _release_context_models(self) -> None:
        for component in (self.brain, self.localizer, self.refiner):
            if component is None:
                continue
            if getattr(component, "model", None) is not None:
                component.model = None
            if getattr(component, "processor", None) is not None:
                component.processor = None
        clear_runtime_memory()

    def _brain_cache_path(self, row: pd.Series) -> Path:
        return self.brain_cache_dir / f"{row['id']}.json"

    def plan_row(
        self,
        row: pd.Series,
        force_refresh: bool = False,
        image: Optional[Image.Image] = None,
        choice_map: Optional[Dict[str, str]] = None,
    ) -> BrainPlan:
        cache_path = self._brain_cache_path(row)
        if cache_path.exists() and not force_refresh:
            return brain_plan_from_json_dict(json.loads(cache_path.read_text(encoding="utf-8")))

        image = image if image is not None else load_image(row["path"])
        choice_map = choice_map if choice_map is not None else build_choices(row, shuffle=False)[0]
        plan = self.brain.plan(image, str(row["question"]), choice_map)
        payload = plan.to_json_dict()
        payload["sample_id"] = str(row["id"])
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return plan

    def prepare_context(
        self,
        row: pd.Series,
        save_panel: bool = True,
        force_refresh: bool = False,
        plan: Optional[BrainPlan] = None,
        force_refresh_plan: bool = False,
    ) -> PreparedContext:
        cache_path = self.cache_dir / f"{row['id']}.json"
        if cache_path.exists() and not force_refresh:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            detections = [BoxCandidate(**item) for item in data.get("detections", [])]
            crops = [CropCandidate(**item) for item in data.get("crops", [])]
            return PreparedContext(
                sample_id=data["sample_id"],
                question_type=data["question_type"],
                focus_objects=data.get("focus_objects", []),
                answer_strategy=data.get("answer_strategy", ""),
                detections=detections,
                crops=crops,
                brain_confidence=float(data.get("brain_confidence", 0.0)),
                grounding_quality=float(data.get("grounding_quality", 0.0)),
                grounding_skip_reason=str(data.get("grounding_skip_reason", "")),
                focus_panel_path=data.get("focus_panel_path"),
            )

        image = load_image(row["path"])
        choice_map, _ = build_choices(row, shuffle=False)
        plan = plan or self.plan_row(
            row,
            force_refresh=force_refresh or force_refresh_plan,
            image=image,
            choice_map=choice_map,
        )
        skip_reason = grounding_skip_reason(plan, choice_map, self.config)
        if skip_reason:
            detections = []
            crops = []
            grounding_quality = 0.0
        else:
            detections = self.localizer.localize(
                image,
                plan.focus_objects,
                plan.question_type,
                choice_map=choice_map,
            )
            crops = self.refiner.refine(image, detections)
            grounding_quality = grounding_quality_score(
                detections,
                image.width,
                image.height,
                plan.question_type,
                self.config,
            )

        focus_panel_path = None
        if save_panel:
            focus_panel = compose_focus_panel(
                image,
                crops,
                tile_size=self.config.panel_tile_size,
                max_items=self.config.max_boxes,
            )
            if focus_panel is not None:
                focus_panel_path = str(self.panel_dir / f"{row['id']}.png")
                focus_panel.save(focus_panel_path)

        context = PreparedContext(
            sample_id=str(row["id"]),
            question_type=plan.question_type,
            focus_objects=plan.focus_objects,
            answer_strategy=plan.answer_strategy,
            detections=detections,
            crops=crops,
            brain_confidence=plan.confidence,
            grounding_quality=grounding_quality,
            grounding_skip_reason=skip_reason or "",
            focus_panel_path=focus_panel_path,
        )
        cache_path.write_text(json.dumps(context.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return context

    def _ensemble_weights(self, question_type: str) -> Dict[str, float]:
        # Combine qtype-specific overrides with each backend's default weight.
        weights = {}
        overrides = QUESTION_TYPE_WEIGHTS.get(question_type, QUESTION_TYPE_WEIGHTS["other"])
        for name, backend in self.backends.items():
            default_weight = backend.model_config.weight
            weights[name] = overrides.get(name, default_weight) * default_weight
        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}

    def predict_row(
        self,
        row: pd.Series,
        context: Optional[PreparedContext] = None,
        force_refresh_context: bool = False,
    ) -> EnsemblePrediction:
        # The student pass always consumes the prepared context, then scores the
        # original image plus any synthesized focus panel.
        context = context or self.prepare_context(row, save_panel=True, force_refresh=force_refresh_context)
        if get_default_device() != "cuda":
            self._release_context_models()
        image = load_image(row["path"])
        focus_panel = None
        if context.focus_panel_path and Path(context.focus_panel_path).exists():
            focus_panel = load_image(context.focus_panel_path)
        student_image = compose_student_image(image, focus_panel)
        student_image = apply_student_sharpen(student_image, context.question_type)

        choice_map, _ = build_choices(row, shuffle=False)
        model_predictions: Dict[str, ModelPrediction] = {}
        for name, backend in self.backends.items():
            model_predictions[name] = backend.score_choices(
                student_image,
                str(row["question"]),
                choice_map,
                context,
            )

        # First fuse the student backends.
        weights = self._ensemble_weights(context.question_type)
        aggregated = {label: 0.0 for label in CHOICE_LABELS}
        for name, prediction in model_predictions.items():
            for label in CHOICE_LABELS:
                aggregated[label] += prediction.choice_probs[label] * weights.get(name, 0.0)

        aggregated = normalize_choice_probs(aggregated)
        # Then blend in small priors from the brain plan and detector outputs.
        for prior_probs, prior_weight in [
            build_brain_choice_prior(context, choice_map, self.config),
            build_detection_choice_prior(context, choice_map, self.config),
            build_detection_count_prior(context, choice_map, self.config),
        ]:
            if prior_probs and prior_weight > 0:
                aggregated = {
                    label: aggregated[label] * (1.0 - prior_weight) + prior_probs[label] * prior_weight
                    for label in CHOICE_LABELS
                }
                aggregated = normalize_choice_probs(aggregated)

        answer = max(aggregated.items(), key=lambda item: item[1])[0]
        confidence = aggregated[answer]
        return EnsemblePrediction(
            answer=answer,
            confidence=confidence,
            aggregated_probs=aggregated,
            model_predictions=model_predictions,
            context=context,
        )

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        output_csv: Optional[str | Path] = None,
        include_debug: bool = False,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            prediction = self.predict_row(row)
            item = {
                "id": row["id"],
                "answer": prediction.answer,
                "confidence": prediction.confidence,
            }
            if include_debug:
                item["question_type"] = prediction.context.question_type
                item["focus_objects"] = "|".join(prediction.context.focus_objects)
                for model_name, model_pred in prediction.model_predictions.items():
                    item[f"{model_name}_answer"] = model_pred.answer
                    item[f"{model_name}_confidence"] = model_pred.confidence
            rows.append(item)

        result = pd.DataFrame(rows)
        if output_csv:
            result.to_csv(output_csv, index=False)
        return result


def load_context_cache(cache_dir: str | Path) -> Dict[str, PreparedContext]:
    cache_map: Dict[str, PreparedContext] = {}
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return cache_map
    for path in cache_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        cache_map[data["sample_id"]] = PreparedContext(
            sample_id=data["sample_id"],
            question_type=data["question_type"],
            focus_objects=data.get("focus_objects", []),
            answer_strategy=data.get("answer_strategy", ""),
            detections=[BoxCandidate(**item) for item in data.get("detections", [])],
            crops=[CropCandidate(**item) for item in data.get("crops", [])],
            brain_confidence=float(data.get("brain_confidence", 0.0)),
            grounding_quality=float(data.get("grounding_quality", 0.0)),
            grounding_skip_reason=str(data.get("grounding_skip_reason", "")),
            focus_panel_path=data.get("focus_panel_path"),
        )
    return cache_map


def load_brain_plan_cache(cache_dir: str | Path) -> Dict[str, BrainPlan]:
    cache_map: Dict[str, BrainPlan] = {}
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return cache_map
    for path in cache_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(data.get("sample_id") or path.stem)
        cache_map[sample_id] = brain_plan_from_json_dict(data)
    return cache_map


class VQAMultistageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: Any,
        train: bool = True,
        context_map: Optional[Dict[str, PreparedContext]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.train = train
        self.context_map = context_map or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image = load_image(row["path"])
        choice_map, remapped_answer = build_choices(row, shuffle=self.train)

        context = self.context_map.get(str(row["id"]))
        focus_panel = None
        extra_context = ""
        if context:
            extra_context = build_context_block(context)
            if context.focus_panel_path and Path(context.focus_panel_path).exists():
                focus_panel = load_image(context.focus_panel_path)
        student_image = compose_student_image(image, focus_panel)
        student_image = apply_student_sharpen(student_image, context.question_type if context else "")
        prompt = build_mc_prompt(str(row["question"]), choice_map, extra_context=extra_context)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": student_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        if self.train and remapped_answer:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": remapped_answer}]})

        return {"messages": messages, "image": student_image}


@dataclass
class DataCollator:
    processor: Any
    train: bool = True

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []
        images = []
        for sample in batch:
            rendered = apply_chat_template_safe(
                self.processor,
                sample["messages"],
                add_generation_prompt=False,
            )
            texts.append(rendered)
            images.append(sample["image"])

        encoded = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        if self.train:
            labels = encoded["input_ids"].clone()
            assistant_ids = self.processor.tokenizer.encode(
                "<|im_start|>assistant",
                add_special_tokens=False,
            )
            for row_idx, label in enumerate(labels):
                tokens = label.tolist()
                for token_idx in range(len(tokens) - len(assistant_ids), -1, -1):
                    if tokens[token_idx : token_idx + len(assistant_ids)] == assistant_ids:
                        labels[row_idx, : token_idx + len(assistant_ids)] = -100
                        break
            encoded["labels"] = labels

        return encoded


def load_qwen_lora_model_for_training(
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    min_pixels: int = 256 * 256,
    max_pixels: int = 512 * 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    quant_4bit: Optional[bool] = None,
) -> Tuple[torch.nn.Module, Any]:
    if any(item is None for item in [LoraConfig, get_peft_model]):
        raise ImportError("peft is required to build the LoRA training model.")
    runtime_device = get_default_device()
    use_4bit = supports_kbit_quantization(runtime_device) if quant_4bit is None else (
        quant_4bit and supports_kbit_quantization(runtime_device)
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        **build_model_load_kwargs(
            device=runtime_device,
            trust_remote_code=True,
            quant_4bit=use_4bit,
        ),
    )
    base_model = finalize_loaded_model(base_model, runtime_device)
    if use_4bit:
        if prepare_model_for_kbit_training is None:
            raise ImportError("peft.prepare_model_for_kbit_training is required for 4-bit LoRA training.")
        base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model = finalize_loaded_model(model, runtime_device)
    return model, processor


def export_context_cache(
    df: pd.DataFrame,
    pipeline: MultiStageVQAPipeline,
    force_refresh: bool = False,
) -> Dict[str, PreparedContext]:
    exported: Dict[str, PreparedContext] = {}
    for _, row in df.iterrows():
        context = pipeline.prepare_context(row, save_panel=True, force_refresh=force_refresh)
        exported[str(row["id"])] = context
    return exported
