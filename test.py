import os
import gc
import json
import subprocess
import time
import uuid
import argparse
import zipfile
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

try:
    from llmcompressor.modifiers.quantization import QuantizationModifier
    QUANTIZATION_MODIFIER_AVAILABLE = True
except ImportError:
    QuantizationModifier = None
    QUANTIZATION_MODIFIER_AVAILABLE = False

from llmcompressor import oneshot


def parse_args():
    p = argparse.ArgumentParser(description="SpQR + Wanda + OmniQuant 파이프라인")
    # 메모리 (24GB GPU OOM 시 --low_memory 사용)
    p.add_argument("--low_memory", action="store_true", help="OOM 방지: num_samples=32, max_len=128, lwc_chunk_size=0(CPU)")
    p.add_argument("--cuda_alloc_conf", type=str, default=None, help="예: expandable_segments:True (OOM 시 메모리 조각화 완화)")
    # GPU 병렬
    p.add_argument("--cuda_devices", type=str, default=None, help="사용할 GPU (예: 0,1,2). 미지정 시 전체 사용")
    p.add_argument("--device_map", type=str, default="auto", help="auto|balanced|sequential|cuda:0 (멀티GPU 시 auto)")
    p.add_argument("--max_memory", type=str, default=None, help="GPU별 메모리 한도 (예: 0:22GiB,1:22GiB)")
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
    p.add_argument("--gems_wanda_scale", type=str, default="sqrt", choices=["none", "sqrt", "log"], help="Gems Wanda 스케일: none|sqrt|log (튀는 값 완화)")
    p.add_argument("--use_symmetric_lwc", action="store_true", default=True, help="대칭 LWC (gamma 1개, zero_point=0)")
    p.add_argument("--no_symmetric_lwc", action="store_false", dest="use_symmetric_lwc", help="비대칭 LWC 사용")

    # LWC 메모리: 0=전부 CPU, 1+=GPU 청크 (OOM 시 0 권장)
    p.add_argument("--lwc_chunk_size", type=int, default=0)
    p.add_argument("--use_8bit_optimizer", action="store_true", help="bitsandbytes 8-bit AdamW (GPU 메모리 절약)")

    # GPTQ / 패킹 (vLLM 호환: 균일 W4A16 필수, mixed precision 미지원)
    p.add_argument("--gptq_block_size", type=int, default=128)
    p.add_argument("--gptq_dampening", type=float, default=0.01)
    p.add_argument(
        "--pack_mode",
        type=str,
        default="gptq",
        choices=["gptq", "rtn"],
        help="gptq: 캘리브 기반 (정확도↑, 시간 소요). rtn: Round-To-Nearest만 (빠름, OmniQuant 결과 보존)",
    )

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="omniquant-spqr")
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--wandb_interval", type=int, default=10)

    # 출력
    p.add_argument("--zip_name", default=None, help="None이면 랜덤 이름 자동 생성 (덮어쓰기 방지)")
    p.add_argument("--zip_dir", default=None, help="zip 저장 폴더 (None=out_dir와 같은 위치의 zips/)")
    p.add_argument("--zip_compress", action="store_true", help="zip 압축 사용 (미지정 시 저장만)")
    p.add_argument("--zip_system", action="store_true", help="시스템 zip 명령 사용 (Mac/Windows 압축해제 호환↑)")
    return p.parse_args()


def main():
    args = parse_args()

    # out_dir를 절대경로로 고정 (실행 위치와 무관하게 동일한 경로에 저장)
    args.out_dir = os.path.abspath(args.out_dir)

    if args.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", args.cuda_alloc_conf)
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        print(f"[INFO] GPU 지정: {args.cuda_devices}")
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
        gems_wanda_scale=args.gems_wanda_scale,
        use_symmetric_lwc=args.use_symmetric_lwc,
        use_reconstruction_loss=args.use_reconstruction_loss,
        lambda_w=args.lambda_w,
        gptq_block_size=args.gptq_block_size,
        gptq_dampening_frac=args.gptq_dampening,
        lwc_chunk_size=args.lwc_chunk_size,
        use_8bit_optimizer=args.use_8bit_optimizer,
        wandb_enable=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        wandb_log_interval=args.wandb_interval,
    )

    _t_start = time.time()
    print(f"[INFO] === SpQR+Wanda+OmniQuant (tau_h={config.tau_h_percentile}, tau_w={config.tau_w_percentile}) ===")
    print(f"[INFO] 저장 경로(절대): {args.out_dir}")
    if args.wandb:
        print(f"[INFO] wandb: {args.wandb_project}")
    print("[INFO] 순정 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if args.max_memory and torch.cuda.is_available():
        max_mem = {}
        for item in args.max_memory.split(","):
            dev, mem = item.strip().split(":")
            max_mem[int(dev)] = mem.strip()
        load_kwargs["max_memory"] = max_mem
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    _n_params = sum(p.numel() for p in model.parameters())
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[INFO] 모델 로드 완료: {_n_params/1e6:.2f}M params ({time.time()-_t_start:.1f}s), GPU={n_gpus}개")
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        dev_counts = {}
        for v in model.hf_device_map.values():
            dev_counts[str(v)] = dev_counts.get(str(v), 0) + 1
        print(f"[INFO] device_map: {dev_counts}")

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
    # ★ vLLM 제출용 균일 W4A16 패키징 (Phase 2)
    # vLLM은 mixed precision(Trash 0/Gems 4/VIP 16) 미지원 → 반드시 균일 INT4로 저장
    # 3-Tier는 OmniQuant 학습 시에만 사용, 최종 출력은 항상 compressed-tensors W4A16
    # =====================================================================
    print("[INFO] 제출용 균일 W4A16 패킹 시작 (compressed-tensors 포맷)...")
    print(f"[INFO] pack_mode={args.pack_mode}, 저장 경로: {args.out_dir}")
    _t3 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # llmcompressor oneshot은 bfloat16과 dtype 충돌 → float32 변환
    if next(model.parameters()).dtype == torch.bfloat16:
        print("[INFO] oneshot 호환을 위해 model bfloat16 → float32 변환")
        model = model.to(torch.float32)

    pack_mode = getattr(args, "pack_mode", "gptq")
    if pack_mode == "rtn" and not QUANTIZATION_MODIFIER_AVAILABLE:
        print("[WARN] QuantizationModifier 없음 — pack_mode=gptq로 전환")
        pack_mode = "gptq"

    try:
        if pack_mode == "rtn":
            # RTN: 캘리브레이션 없이 가중치만 Round-To-Nearest 패킹 (빠름, OmniQuant 결과 보존)
            recipe = [
                QuantizationModifier(
                    targets=["Linear"],
                    ignore=["embed_tokens", "lm_head"],
                    scheme="W4A16",
                )
            ]
            try:
                oneshot(model=model, recipe=recipe, pipeline="datafree")
            except (TypeError, ValueError) as e1:
                print(f"[WARN] RTN datafree 실패({e1}), 최소 캘리브로 재시도...")
                ds_rtn = tokenized_ds.select(range(1))
                oneshot(
                    model=model,
                    dataset=ds_rtn,
                    recipe=recipe,
                    max_seq_length=args.max_len,
                    num_calibration_samples=1,
                )
        else:
            # GPTQ: 캘리브레이션 기반 (정확도↑, 시간 소요)
            recipe = [
                GPTQModifier(
                    targets=["Linear"],
                    ignore=["embed_tokens", "lm_head"],
                    scheme="W4A16",
                    block_size=config.gptq_block_size,
                    dampening_frac=config.gptq_dampening_frac,
                    actorder="static",
                )
            ]
            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=args.max_len,
                num_calibration_samples=args.num_samples,
            )
    except Exception as e:
        if pack_mode == "rtn":
            print(f"[WARN] RTN 패킹 실패: {e}")
            print("[INFO] GPTQ로 폴백합니다...")
            recipe = [
                GPTQModifier(
                    targets=["Linear"],
                    ignore=["embed_tokens", "lm_head"],
                    scheme="W4A16",
                    block_size=config.gptq_block_size,
                    dampening_frac=config.gptq_dampening_frac,
                    actorder="static",
                )
            ]
            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=args.max_len,
                num_calibration_samples=args.num_samples,
            )
        else:
            print(f"[ERROR] oneshot 실패: {e}")
            fallback_dir = os.path.join(os.path.dirname(args.out_dir), "model_pre_gptq")
            os.makedirs(fallback_dir, exist_ok=True)
            model.save_pretrained(fallback_dir)
            tokenizer.save_pretrained(fallback_dir)
            raise RuntimeError(f"oneshot 실패. 에러: {e}") from e

    print(f"[INFO] 패킹 완료 ({time.time()-_t3:.1f}s)")

    print("[INFO] 모델 저장 준비 (quantization_config 주입)...")
    # 주의: model.to(bfloat16) 금지! oneshot이 만든 QuantLinear의 int8 패킹 가중치가 손상됨 → vLLM INT4 커널 미동작
    # oneshot 출력을 그대로 저장해야 4비트 패킹이 유지됨

    # vLLM 인식 + save_compressed 비트패킹 유도: quantization_config 수동 주입
    model.config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": config.gptq_block_size,
                },
            }
        },
        "ignore": ["lm_head", "embed_tokens"],
    }

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
    zip_dir = os.path.abspath(args.zip_dir) if args.zip_dir else os.path.join(os.path.dirname(out_dir_abs), "zips")
    os.makedirs(zip_dir, exist_ok=True)
    zip_name = args.zip_name or f"submit_{uuid.uuid4().hex[:12]}"
    zip_path = os.path.join(zip_dir, f"{zip_name}.zip")
    print(f"[INFO] {zip_name}.zip 생성 중... (경로: {zip_path})")
    zip_compress = getattr(args, "zip_compress", False)
    zip_system = getattr(args, "zip_system", False)
    try:
        if zip_system and shutil.which("zip"):
            # 시스템 zip: Mac/Windows 아카이버 호환성 ↑
            parent_dir = os.path.dirname(out_dir_abs)
            model_dir_name = os.path.basename(out_dir_abs)
            cwd = os.getcwd()
            os.chdir(parent_dir)
            try:
                cmd = ["zip", "-r", "-0" if not zip_compress else "-9", zip_path, model_dir_name]
                subprocess.run(cmd, check=True)
            finally:
                os.chdir(cwd)
            print(f"[INFO] zip 모드: 시스템 zip ({'압축' if zip_compress else '저장만'})")
        else:
            if zip_system and not shutil.which("zip"):
                print("[WARN] 시스템 zip 미설치, Python zipfile 사용")
            compression = zipfile.ZIP_DEFLATED if zip_compress else zipfile.ZIP_STORED
            with zipfile.ZipFile(zip_path, "w", compression, allowZip64=True) as zf:
                for root, _, files in os.walk(out_dir_abs):
                    for f in files:
                        full_path = os.path.join(root, f)
                        arcname = os.path.join(os.path.basename(out_dir_abs), os.path.relpath(full_path, out_dir_abs))
                        zf.write(full_path, arcname=arcname)
            print(f"[INFO] zip 모드: Python zipfile ({'압축' if zip_compress else '저장만'})")
        _zip_size_mb = os.path.getsize(zip_path) / 1024 / 1024
        print(f"[INFO] 생성 완료: {zip_path} ({_zip_size_mb:.1f} MB)")
    except Exception as e:
        print(f"[ERROR] zip 생성 실패: {e}")
        raise
    if not os.path.isfile(zip_path) or os.path.getsize(zip_path) == 0:
        raise RuntimeError(f"zip 파일이 생성되지 않았습니다: {zip_path}")

    _total = time.time() - _t_start
    print(f"\n[INFO] === 전체 파이프라인 완료 ({_total:.1f}s) ===")


if __name__ == "__main__":
    main()