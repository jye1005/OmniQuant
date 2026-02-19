import os
import gc
import json
import time
import argparse
import torch
import shutil
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_omniquant import (
    SpQRWandaConfig,
    calculate_hessian_wanda,
    apply_pruning_and_bias_correction,
    run_3tier_omniquant,
)
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot


def parse_args():
    p = argparse.ArgumentParser(description="SpQR + Wanda + OmniQuant 파이프라인")
    # 메모리 (24GB GPU OOM 시 --low_memory 사용)
    p.add_argument("--low_memory", action="store_true", help="OOM 방지: num_samples=32, max_len=128, lwc_chunk_size=0(CPU)")
    p.add_argument("--cuda_alloc_conf", type=str, default=None, help="예: expandable_segments:True (OOM 시 메모리 조각화 완화)")
    # 데이터/모델
    p.add_argument("--model_id", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    p.add_argument("--out_dir", default="./model")
    p.add_argument("--dataset_id", default="LGAI-EXAONE/MANTA-1M")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--calib_batch_size", type=int, default=1)

    # SpQR/Wanda
    p.add_argument("--tau_h", type=float, default=0.30)
    p.add_argument("--tau_w", type=float, default=0.50)
    p.add_argument("--bias_correction", action="store_true", default=True)
    p.add_argument("--no_bias_correction", action="store_false", dest="bias_correction")

    # OmniQuant LWC
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--use_reconstruction_loss", action="store_true", help="MSE loss 사용 (메모리 2배)")
    p.add_argument("--lambda_w", type=float, default=0.01)
    p.add_argument("--vip_penalty", type=float, default=100.0)
    p.add_argument("--gems_penalty", type=float, default=0.01)

    # LWC 메모리: 0=전부 CPU, 1+=GPU 청크 (OOM 시 0 권장)
    p.add_argument("--lwc_chunk_size", type=int, default=0)

    # GPTQ
    p.add_argument("--gptq_block_size", type=int, default=128)
    p.add_argument("--gptq_dampening", type=float, default=0.01)

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="omniquant-spqr")
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--wandb_interval", type=int, default=10)

    # 출력
    p.add_argument("--zip_name", default="baseline_submit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", args.cuda_alloc_conf)
    if args.low_memory:
        args.num_samples = 32
        args.max_len = 128
        args.lwc_chunk_size = 0
        print("[INFO] --low_memory: num_samples=32, max_len=128, lwc_chunk_size=0 (CPU)")

    config = SpQRWandaConfig(
        tau_h_percentile=args.tau_h,
        tau_w_percentile=args.tau_w,
        bias_correction=args.bias_correction,
        omniquant_epochs=args.epochs,
        omniquant_lr=args.lr,
        vip_penalty_weight=args.vip_penalty,
        gems_penalty_weight=args.gems_penalty,
        use_reconstruction_loss=args.use_reconstruction_loss,
        lambda_w=args.lambda_w,
        gptq_block_size=args.gptq_block_size,
        gptq_dampening_frac=args.gptq_dampening,
        lwc_chunk_size=args.lwc_chunk_size,
        wandb_enable=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        wandb_log_interval=args.wandb_interval,
    )

    _t_start = time.time()
    print(f"[INFO] === SpQR+Wanda+OmniQuant (tau_h={config.tau_h_percentile}, tau_w={config.tau_w_percentile}) ===")
    if args.wandb:
        print(f"[INFO] wandb: {args.wandb_project}")
    print("[INFO] 순정 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 모델 로드 완료: {_n_params/1e6:.2f}M params ({time.time()-_t_start:.1f}s)")

    print("[INFO] 캘리브레이션 데이터 준비 중...")
    ds = load_dataset(args.dataset_id, split=f"{args.dataset_split}[:{args.num_samples}]")

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

    ds = ds.map(preprocess, remove_columns=["conversations"] if "conversations" in ds.column_names else ds.column_names)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_len,
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
        batch_size=args.calib_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"[INFO] 캘리브레이션: {len(tokenized_ds)} 샘플, batch={args.calib_batch_size}, max_len={args.max_len}")

    # =====================================================================
    # ★ 지혜 님만의 독창적인 압축 최적화 구간 (Phase 1)
    # =====================================================================
    print("[INFO] 커스텀 알고리즘: Hessian & Wanda 스코어 계산...")
    _t1 = time.time()
    masks, wanda_scores, x_means = calculate_hessian_wanda(model, calib_dataloader, config=config)

    print("[INFO] SpQR Pruning + Bias Correction 적용...")
    apply_pruning_and_bias_correction(model, masks, x_means, config=config)
    print(f"[INFO] Phase 1 완료 ({time.time()-_t1:.1f}s)")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[INFO] 커스텀 알고리즘: 3-Tier OmniQuant 학습 시작...")
    _t2 = time.time()
    model = run_3tier_omniquant(
        model, calib_dataloader, masks, wanda_scores, config=config
    )
    print(f"[INFO] OmniQuant 학습 완료 ({time.time()-_t2:.1f}s)")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # ★ vLLM 제출용 INT4 패키징 구간 (Phase 2)
    # =====================================================================
    print("[INFO] 제출용 INT4 포장(Packing) 작업 시작...")
    _t3 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    recipe = [
        GPTQModifier(
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
            scheme="W4A16",
            block_size=config.gptq_block_size,
            dampening_frac=config.gptq_dampening_frac,
            actorder=False,
        )
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_len,
        num_calibration_samples=args.num_samples,
    )
    print(f"[INFO] GPTQ 포장 완료 ({time.time()-_t3:.1f}s)")

    print("[INFO] 모델 저장 중...")
    model.save_pretrained(args.out_dir, save_compressed=True)
    tokenizer.save_pretrained(args.out_dir)
    n_files = len([f for f in os.scandir(args.out_dir) if f.is_file()])
    _out_size_mb = sum(f.stat().st_size for f in os.scandir(args.out_dir) if f.is_file()) / 1024 / 1024
    _safetensors_mb = sum(
        f.stat().st_size for f in os.scandir(args.out_dir)
        if f.is_file() and f.name.endswith(".safetensors")
    ) / 1024 / 1024
    print(f"[INFO] 모델 저장 완료: {args.out_dir} ({n_files}개 파일, {_out_size_mb:.1f} MB)")
    if n_files == 0:
        raise RuntimeError(f"저장 실패: {args.out_dir}에 파일이 없습니다. oneshot/save_pretrained 확인 필요.")
    if _safetensors_mb > 1500:
        print(f"[WARN] ⚠️ .safetensors 합계 {_safetensors_mb:.0f}MB — INT4면 ~800MB여야 함. save_compressed 미적용 가능성 있음.")
    config_path = os.path.join(args.out_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
        qc_str = json.dumps(qc)
        if "compressed-tensors" not in qc_str and "quant_method" not in qc_str:
            print("[WARN] ⚠️ config.json에 quant_method/compressed-tensors 없음 — vLLM이 인식 못할 수 있음.")

    # =====================================================================
    # ★ 압축 및 제출 준비 (Phase 3)
    # =====================================================================
    out_dir_abs = os.path.abspath(args.out_dir)
    zip_base = os.path.join(os.path.dirname(out_dir_abs), args.zip_name)
    zip_path = f"{zip_base}.zip"
    print(f"[INFO] {args.zip_name}.zip 생성 중... (경로: {zip_path})")
    shutil.make_archive(base_name=zip_base, format="zip", root_dir=os.path.dirname(out_dir_abs), base_dir=os.path.basename(out_dir_abs))
    _zip_size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"[INFO] 생성 완료: {zip_path} ({_zip_size_mb:.1f} MB)")

    _total = time.time() - _t_start
    print(f"\n[INFO] === 전체 파이프라인 완료 ({_total:.1f}s) ===")


if __name__ == "__main__":
    main()