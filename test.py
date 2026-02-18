import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 지혜 님의 커스텀 로직 불러오기
from custom_omniquant import calculate_hessian_wanda, run_3tier_omniquant

# 2. 포장(Packing)을 위해 베이스라인 라이브러리 사용
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B" # 반드시 순정 모델 사용!
OUT_DIR = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

print("[INFO] 순정 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

print("[INFO] 캘리브레이션 데이터 준비 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}
ds = ds.map(preprocess)

# (주의: custom_omniquant.py에서 사용할 수 있도록 ds를 DataLoader 형태로 변환하는 코드가 필요합니다)
calib_dataloader = ds # 임시 표현

# =====================================================================
# ★ 지혜 님만의 독창적인 압축 최적화 구간 (Phase 1)
# =====================================================================
print("[INFO] 커스텀 알고리즘: Hessian & Wanda 스코어 계산...")
masks, wanda_scores = calculate_hessian_wanda(model, calib_dataloader)

print("[INFO] 커스텀 알고리즘: 3-Tier OmniQuant 학습 시작...")
model = run_3tier_omniquant(model, calib_dataloader, masks, wanda_scores)

# 여기까지 오면 model의 가중치는 이미 지혜 님의 의도대로 완벽하게 깎인 상태입니다.
# 이제 이 깎인 값을 "고정"시키고 4비트로 포장만 하면 됩니다.

# =====================================================================
# ★ vLLM 제출용 INT4 패키징 구간 (Phase 2)
# =====================================================================
print("[INFO] 제출용 INT4 포장(Packing) 작업 시작...")

# llmcompressor에게 "우리가 이미 다 깎아놨으니까, 너는 스케일만 잡고 INT4로 포장만 해!" 라고 지시합니다.
recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        scheme="W4A16",
        block_size=128,
        dampening_frac=0.01,
        # 중요: actorder를 비활성화해야 커스텀 가중치가 덜 망가집니다.
        actorder=False 
    )
]

# 여기서 oneshot을 돌리면, llmcompressor는 지혜 님이 예쁘게 다듬어 놓은 가중치(Fake Quant)를 
# 입력으로 받아서 INT4 규격(Real Quant)으로 꾹꾹 눌러 담아줍니다.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] 포장 완료, 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)

# 핵심: save_compressed=True 가 INT32 텐서로 변환하여 저장해 줍니다.
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

# =====================================================================
# ★ 압축 및 제출 준비 (Phase 3)
# =====================================================================
zip_name = "baseline_submit"
print(f"[INFO] {zip_name}.zip 생성 중...")
shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=OUT_DIR)
print(f"[INFO] 생성 완료: {zip_name}.zip")