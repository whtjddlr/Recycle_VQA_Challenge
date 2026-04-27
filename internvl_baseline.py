import torch
import os, random, re
import pandas as pd
import traceback
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm

# ==========================================
# 1. Environment & Hardware Configuration
# ==========================================
MODEL_ID = "OpenGVLab/InternVL2_5-4B"
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TRAIN_CSV = "train.csv"
VALID_CSV = "test.csv"
SAVE_PATH = "./best_internvl_ensemble"
OUTPUT_CSV = "submission_internvl_ensemble_ready.csv"

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 8e-5
ACCUM_STEPS = 4

print("[System] 앙상블 파이프라인 초기화 시작 (4-bit 양자화 및 정밀 확률 추출 적용)")

# ==========================================
# 2. Model Loading (4-bit Quantization) & Patching
# ==========================================
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModel.from_pretrained(
    MODEL_ID, 
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True, 
    device_map="auto"
)

# [Patch] 모델 순전파(forward) 시 inputs_embeds 충돌 방지 패치 적용
original_forward = base_model.forward
def patched_forward(*args, **kwargs):
    kwargs.pop("inputs_embeds", None)
    return original_forward(*args, **kwargs)
base_model.forward = patched_forward

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# <IMG_CONTEXT> 토큰 설정
img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
base_model.img_context_token_id = img_context_token_id
if hasattr(base_model, 'config'): 
    base_model.config.img_context_token_id = img_context_token_id
    base_model.config.use_cache = False

# ==========================================
# 3. LoRA Configuration & Checkpoint Resume
# ==========================================
target_modules = [n for n, m in base_model.named_modules() if isinstance(m, torch.nn.Linear) and any(k in n for k in ["qkv", "proj", "fc1", "fc2", "wqkv", "wo", "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"])]
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

start_epoch = 0
latest_ckpt = None

# 체크포인트 탐색 (가장 최근에 정상 저장된 Epoch 확인)
for i in range(EPOCHS, 0, -1):
    ckpt_dir = f"{SAVE_PATH}_ep{i}"
    if os.path.isdir(ckpt_dir) and (os.path.exists(f"{ckpt_dir}/adapter_model.bin") or os.path.exists(f"{ckpt_dir}/adapter_model.safetensors")):
        start_epoch = i
        latest_ckpt = ckpt_dir
        break

if latest_ckpt:
    print(f"[Resume] 기존 체크포인트({latest_ckpt}) 감지됨. Epoch {start_epoch + 1}부터 학습을 재개합니다.")
    model = PeftModel.from_pretrained(base_model, latest_ckpt, is_trainable=True)
else:
    print("[New] 기존 체크포인트가 존재하지 않아 Epoch 1부터 학습을 시작합니다.")
    model = get_peft_model(base_model, lora_config)

model.enable_input_require_grads()

# ==========================================
# 4. Image Preprocessing & Dataset Construction
# ==========================================
def prepare_image_vision20(image):
    """비율 왜곡 방지를 위한 레터박스(Letterbox) 패딩 적용"""
    w, h = image.size
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_img.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    return new_img

# 다단계 CoT(Chain of Thought) 시스템 프롬프트 설정
SYSTEM_INSTRUCT = (
    "당신은 쓰레기 분리배출, 재질 분석 및 객체 카운팅에 특화된 최고 수준의 시각 질의응답(VQA) 전문가입니다.\n"
    "특히 '특정 재질 조건이 붙은 개수 세기(예: 플라스틱 컵 개수, 종이 상자 개수)'에 매우 주의해야 합니다.\n"
    "다음 3단계를 머릿속으로 철저히 분석한 뒤 최종 정답을 고르세요:\n"
    "1. [대상 포착] 숨겨지거나 겹쳐있는 물체(비닐 속, 상자 뒤 등), 뚜껑/빨대의 유무와 색상을 모두 파악합니다.\n"
    "2. [재질 판별] 포착한 물체들의 재질(투명 플라스틱, 불투명 플라스틱, 유리, 캔, 종이 등)을 명확히 구분합니다.\n"
    "3. [조건 카운팅] 질문에서 요구한 '재질'과 '품목'에 정확히 일치하는 객체만 개수를 셉니다.\n"
    "[경고] 성급하게 결론 내리지 말고 (a), (b), (c), (d) 보기를 모두 비교하세요.\n"
    "어떠한 설명도 덧붙이지 말고, 반드시 a, b, c, d 중 가장 정확한 정답 하나만 소문자 알파벳 한 글자로 출력하세요."
)

class VQADataset(Dataset):
    def __init__(self, df, img_processor, tokenizer):
        self.df = df.reset_index(drop=True)
        self.img_processor = img_processor
        self.tokenizer = tokenizer

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        
        raw_img = Image.open(row["path"]).convert("RGB")
        padded_img = prepare_image_vision20(raw_img)
        
        pv = self.img_processor(images=padded_img, return_tensors="pt").pixel_values.to(torch.bfloat16)
        if pv.dim() == 5: pv = pv.squeeze(0)
        num_patches = pv.size(0) 
        image_prompt = f"<img>{'<IMG_CONTEXT>' * 256 * num_patches}</img>"
        
        options = [('a', str(row['a'])), ('b', str(row['b'])), ('c', str(row['c'])), ('d', str(row['d']))]
        orig_ans_key = str(row['answer']).lower().strip()
        correct_text = next((text for key, text in options if key == orig_ans_key), "")
        
        # 모델의 위치 편향 학습을 방지하기 위한 보기 순서 무작위 섞기
        random.shuffle(options)
        sh_str = ""
        new_ans_key = ""
        for idx, new_label in enumerate(['a', 'b', 'c', 'd']):
            sh_str += f"({new_label}) {options[idx][1]}\n"
            if options[idx][1] == correct_text: 
                new_ans_key = new_label 
            
        q_text = f"{row['question']}\n{sh_str}정답:"
        prompt = f"{SYSTEM_INSTRUCT}\n\nUser: {image_prompt}\n{q_text}\nAssistant: {new_ans_key}"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        
        return {
            "pixel_values": pv, 
            "image_flags": torch.ones(num_patches, dtype=torch.long),
            "input_ids": inputs["input_ids"].squeeze(0), 
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0).clone()
        }

def collate_fn(batch):
    pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return {
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0),
        "image_flags": torch.cat([x["image_flags"] for x in batch], dim=0), 
        "input_ids": torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=pad_val),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence([x["labels"] for x in batch], batch_first=True, padding_value=-100)
    }

# ==========================================
# 5. Confidence Extraction Module
# ==========================================
option_token_map = {opt: tokenizer.encode(opt, add_special_tokens=False) for opt in ["a", "b", "c", "d"]}

def compute_ensemble_probs(first_step_scores):
    """모델의 출력 Logit을 바탕으로 앙상블을 위한 각 보기별 Softmax 확률을 계산합니다."""
    option_logits = {}
    for opt in ["a", "b", "c", "d"]:
        ids = option_token_map.get(opt, [])
        if ids:
            vals = [float(first_step_scores[i].item()) for i in ids]
            option_logits[opt] = max(vals)
        else:
            option_logits[opt] = -1e9
            
    logits = torch.tensor([option_logits["a"], option_logits["b"], option_logits["c"], option_logits["d"]])
    probs = torch.softmax(logits, dim=0).tolist()
    prob_map = {"a": probs[0], "b": probs[1], "c": probs[2], "d": probs[3]}
    
    ranked = sorted(prob_map.items(), key=lambda x: -x[1])
    best_opt = ranked[0][0]
    confidence = float(ranked[0][1])
    
    return best_opt, confidence, prob_map

# ==========================================
# 6. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    
    # ------------------------------------------
    # [PHASE 1] Model Fine-tuning
    # ------------------------------------------
    if start_epoch < EPOCHS:
        if not os.path.exists(TRAIN_CSV):
            print(f"[Error] {TRAIN_CSV} 파일을 찾을 수 없습니다. 프로세스를 종료합니다.")
            exit()
            
        train_df = pd.read_csv(TRAIN_CSV)
        dataset = VQADataset(train_df, image_processor, tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        remaining_epochs = EPOCHS - start_epoch
        total_steps = (len(loader) // ACCUM_STEPS) * remaining_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

        print(f"[PHASE 1] 모델 파인튜닝 시작 (남은 Epoch: {remaining_epochs}, LR: {LEARNING_RATE})")
        model.train()
        
        for epoch in range(start_epoch, EPOCHS):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            optimizer.zero_grad()
            
            for step, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / ACCUM_STEPS
                loss.backward()
                
                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                if step % 10 == 0:
                    pbar.set_postfix({'loss': f"{outputs.loss.item() * ACCUM_STEPS:.4f}"})
                    
            # Epoch 체크포인트 저장
            model.save_pretrained(f"{SAVE_PATH}_ep{epoch+1}")
            torch.cuda.empty_cache() 

        # 최종 모델 저장
        model.save_pretrained(SAVE_PATH)
        print(f"[Save] 최종 모델 가중치 저장 완료: {SAVE_PATH}")
    else:
        print("[Info] 지정된 Epoch의 학습이 이미 완료되었습니다. 추론 단계를 진행합니다.")

    # ------------------------------------------
    # [PHASE 2] Inference & Ensemble Preparation
    # ------------------------------------------
    model.eval()
    
    # 래퍼 모델을 해제하여 베이스 모델의 추론 메서드(generate)를 정상적으로 호출하기 위한 설정
    vlm_model = model.base_model.model if hasattr(model, "base_model") else model
    vlm_model.img_context_token_id = img_context_token_id

    if not os.path.exists(VALID_CSV):
        print(f"[Error] {VALID_CSV} 파일을 찾을 수 없습니다. 프로세스를 종료합니다.")
        exit()

    test_df = pd.read_csv(VALID_CSV)
    detailed_results = []
    
    print(f"[PHASE 2] 테스트 데이터 {len(test_df)}건에 대한 추론 및 앙상블 확률 추출 시작")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        try:
            raw_img = Image.open(row["path"]).convert("RGB")
            padded_img = prepare_image_vision20(raw_img)
            pv = image_processor(images=padded_img, return_tensors="pt").pixel_values.to(torch.bfloat16).to(device)
            if pv.dim() == 5: 
                pv = pv.squeeze(0)
            
            num_patches = pv.size(0)
            image_prompt = f"<img>{'<IMG_CONTEXT>' * 256 * num_patches}</img>"
            
            q_text = f"{row['question']}\n(a) {row['a']}\n(b) {row['b']}\n(c) {row['c']}\n(d) {row['d']}\n정답:"
            prompt = f"{SYSTEM_INSTRUCT}\n\nUser: {image_prompt}\n{q_text}\nAssistant: "
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = vlm_model.generate(
                    pixel_values=pv,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=2,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False
                )
                
            # 앙상블 분석을 위한 개별 확률값 추출
            best_opt, conf, prob_map = compute_ensemble_probs(outputs.scores[0][0])
            
            detailed_results.append({
                "id": row["id"],
                "answer": best_opt,
                "confidence": round(conf, 4),
                "prob_a": round(prob_map["a"], 4),
                "prob_b": round(prob_map["b"], 4),
                "prob_c": round(prob_map["c"], 4),
                "prob_d": round(prob_map["d"], 4)
            })

        except Exception as e:
            print(f"[Error] 처리 중 예외 발생 (ID {row['id']}): {e}")
            traceback.print_exc()
            # 예외 발생 시 기본값으로 데이터 누락 방지
            detailed_results.append({
                "id": row["id"], "answer": "a", "confidence": 0.0, 
                "prob_a": 1.0, "prob_b": 0.0, "prob_c": 0.0, "prob_d": 0.0
            })

    # 최종 제출 및 앙상블 전용 파일 저장
    pd.DataFrame(detailed_results).to_csv(OUTPUT_CSV, index=False)
    print(f"[Complete] 모든 파이프라인 정상 종료. 결과 파일 생성 완료: {OUTPUT_CSV}")