import os
import time
import torch
import shutil
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 지혜 님의 커스텀 로직 불러오기
from custom_omniquant import (
    SpQRWandaConfig,
    calculate_hessian_wanda,
    apply_pruning_and_bias_correction,
    run_3tier_omniquant,
)

# 2. 포장(Packing)을 위해 베이스라인 라이브러리 사용
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B" # 반드시 순정 모델 사용!
OUT_DIR = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# ========== SpQR + Wanda + OmniQuant 하이퍼파라미터 (여기서 지정) ==========
SPQR_CONFIG = SpQRWandaConfig(
    # [1단계] Hessian Trash (Pruning)
    tau_h_percentile=0.30,       # 하위 30% → Trash (0-bit)
    # [2단계] Wanda Gems vs VIP
    tau_w_percentile=0.50,       # 상위 50% → VIP
    spqr_blocksize=16,
    spqr_outlier_percentile=None,
    n_bits_gems=4,
    n_bits_vip=16,
    bias_correction=True,  # SpQR Bias Correction (Pruning 출력 보정)
    # OmniQuant LWC
    omniquant_epochs=20,
    omniquant_lr=1e-2,
    vip_penalty_weight=100.0,
    gems_penalty_weight=0.01,
    use_reconstruction_loss=True,  # MSE(original_out, quant_out) + wanda_penalty
    lambda_w=0.01,
    # GPTQ 포장
    gptq_block_size=128,
    gptq_dampening_frac=0.01,
    # wandb (실험 추적)
    wandb_enable=True,  # True로 설정 시 wandb 로깅
    wandb_project="omniquant-spqr",
    wandb_run_name=None,  # None이면 자동, 또는 "exaone-1.2b-exp1" 등 지정
    wandb_log_interval=10,  # 10 batch마다 로깅
)

_t_start = time.time()
print(f"[INFO] === SpQR+Wanda+OmniQuant 파이프라인 (tau_h={SPQR_CONFIG.tau_h_percentile}, tau_w={SPQR_CONFIG.tau_w_percentile}) ===")
print("[INFO] 순정 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
_n_params = sum(p.numel() for p in model.parameters())
print(f"[INFO] 모델 로드 완료: {_n_params/1e6:.2f}M params ({time.time()-_t_start:.1f}s)")

print("[INFO] 캘리브레이션 데이터 준비 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}


ds = ds.map(preprocess, remove_columns=["conversations"] if "conversations" in ds.column_names else ds.column_names)


def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length",
    )


tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(b["input_ids"]) for b in batch]),
        "attention_mask": torch.stack([torch.tensor(b["attention_mask"]) for b in batch]),
    }


calib_dataloader = DataLoader(
    tokenized_ds,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)
print(f"[INFO] 캘리브레이션: {len(tokenized_ds)} 샘플, batch_size=4, max_len={MAX_SEQUENCE_LENGTH}")

# =====================================================================
# ★ 지혜 님만의 독창적인 압축 최적화 구간 (Phase 1)
# =====================================================================
print("[INFO] 커스텀 알고리즘: Hessian & Wanda 스코어 계산...")
_t1 = time.time()
masks, wanda_scores, x_means = calculate_hessian_wanda(model, calib_dataloader, config=SPQR_CONFIG)

print("[INFO] SpQR Pruning + Bias Correction 적용...")
apply_pruning_and_bias_correction(model, masks, x_means, config=SPQR_CONFIG)
print(f"[INFO] Phase 1 완료 ({time.time()-_t1:.1f}s)")

print("[INFO] 커스텀 알고리즘: 3-Tier OmniQuant 학습 시작...")
_t2 = time.time()
model = run_3tier_omniquant(
    model, calib_dataloader, masks, wanda_scores, config=SPQR_CONFIG
)
print(f"[INFO] OmniQuant 학습 완료 ({time.time()-_t2:.1f}s)")

# 여기까지 오면 model의 가중치는 이미 지혜 님의 의도대로 완벽하게 깎인 상태입니다.
# 이제 이 깎인 값을 "고정"시키고 4비트로 포장만 하면 됩니다.

# =====================================================================
# ★ vLLM 제출용 INT4 패키징 구간 (Phase 2)
# =====================================================================
print("[INFO] 제출용 INT4 포장(Packing) 작업 시작...")
_t3 = time.time()

# llmcompressor에게 "우리가 이미 다 깎아놨으니까, 너는 스케일만 잡고 INT4로 포장만 해!" 라고 지시합니다.
recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        scheme="W4A16",
        block_size=SPQR_CONFIG.gptq_block_size,
        dampening_frac=SPQR_CONFIG.gptq_dampening_frac,
        actorder=False,  # 커스텀 가중치 보존
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
print(f"[INFO] GPTQ 포장 완료 ({time.time()-_t3:.1f}s)")

print("[INFO] 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)

# 핵심: save_compressed=True 가 INT32 텐서로 변환하여 저장해 줍니다.
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
_out_size_mb = sum(f.stat().st_size for f in os.scandir(OUT_DIR) if f.is_file()) / 1024 / 1024
print(f"[INFO] 모델 저장 완료: {OUT_DIR} ({_out_size_mb:.1f} MB)")

# =====================================================================
# ★ 압축 및 제출 준비 (Phase 3)
# =====================================================================
zip_name = "baseline_submit"
print(f"[INFO] {zip_name}.zip 생성 중...")
shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=OUT_DIR)
_zip_size_mb = os.path.getsize(f"{zip_name}.zip") / 1024 / 1024
print(f"[INFO] 생성 완료: {zip_name}.zip ({_zip_size_mb:.1f} MB)")

_total = time.time() - _t_start
print(f"\n[INFO] === 전체 파이프라인 완료 ({_total:.1f}s) ===")