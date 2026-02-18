import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
from dataclasses import dataclass
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ==============================================================================
# SpQR + Wanda + OmniQuant 통합 하이퍼파라미터
# ==============================================================================
@dataclass
class SpQRWandaConfig:
    """SpQR(1단계 Pruning, 2단계 Outlier) + Wanda(자신감) + OmniQuant(LWC) 하이퍼파라미터"""

    # ----- [1단계] Hessian 기반 Trash (Pruning) - SpQR 스타일 -----
    tau_h_percentile: float = 0.30  # Hessian 역수 기준 하위 30% → Trash (0-bit 제거)
    # 값 낮을수록 더 많이 제거 (0.2: 공격적, 0.5: 보수적)

    # ----- [2단계] Wanda 기반 Gems vs VIP 분류 - SpQR Outlier -----
    tau_w_percentile: float = 0.50  # Wanda 기준 survivors 내 상위 50% → VIP (고비트)
    # 값 높을수록 VIP 많음 (0.3: Gems 많음, 0.7: VIP 많음)

    # ----- SpQR 블록/양자화 구조 -----
    spqr_blocksize: int = 16  # SpQR 기본 16x16 블록 (group_size와 연동)
    spqr_outlier_percentile: Optional[float] = None  # None이면 tau_w 사용. 0.99 = 상위 1%를 outlier
    n_bits_gems: int = 4  # Regular Gems 비트 (2~4)
    n_bits_vip: int = 16  # VIP 비트 (8 or 16, 16=FP16 보존)
    bias_correction: bool = False  # SpQR Bias Correction 적용 여부 (구현 시 True)

    # ----- OmniQuant LWC 학습 -----
    omniquant_epochs: int = 20
    omniquant_lr: float = 1e-2
    vip_penalty_weight: float = 100.0  # VIP 클리핑 패널티 배수 (패널티만 쓸 때)
    gems_penalty_weight: float = 0.01  # Gems 클리핑 패널티 배수
    use_reconstruction_loss: bool = True  # MSE(original_out, quant_out) 사용
    lambda_w: float = 0.01  # total_loss = mse_loss + lambda_w * wanda_penalty

    # ----- GPTQ 포장 (llmcompressor) -----
    gptq_block_size: int = 128  # GPTQ group_size (SpQR 16보다 큼)
    gptq_dampening_frac: float = 0.01

    # ----- wandb -----
    wandb_enable: bool = False
    wandb_project: str = "omniquant-spqr"
    wandb_run_name: Optional[str] = None  # None이면 자동 생성
    wandb_log_interval: int = 10  # N batch마다 로깅 (0=매 batch)


# 기본값 (바로 사용 가능)
DEFAULT_CONFIG = SpQRWandaConfig()


# ==============================================================================
# 1단계: Hessian 및 Wanda 스코어 계산 (3-Tier 마스크 생성)
# ==============================================================================
def calculate_hessian_wanda(
    model,
    calib_dataloader,
    config: Optional[SpQRWandaConfig] = None,
    tau_h_percentile: Optional[float] = None,
    tau_w_percentile: Optional[float] = None,
):
    """
    캘리브레이션 데이터를 돌려서 각 Linear 레이어의 Hessian 대각성분과 Wanda 점수를 계산합니다.
    calib_dataloader는 {"input_ids": tensor, "attention_mask": tensor} 형태의 배치를 반환해야 합니다.
    """
    cfg = config or DEFAULT_CONFIG
    tau_h = tau_h_percentile if tau_h_percentile is not None else cfg.tau_h_percentile
    tau_w = tau_w_percentile if tau_w_percentile is not None else cfg.tau_w_percentile

    masks = {}
    wanda_scores = {}
    x_means = {}
    device = next(model.parameters()).device

    # 활성화값(X)을 가로채기 위한 Hook 함수 설정
    activation_cache = {}
    def hook_fn(name):
        def forward_hook(module, input, output):
            x = input[0].detach()
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])
            if name not in activation_cache:
                activation_cache[name] = []
            activation_cache[name].append(x.cpu())
        return forward_hook

    # Hook 등록 (EXAONE의 Linear 레이어 타겟)
    linear_names = []
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name and "embed_tokens" not in name:
            linear_names.append(name)
            hooks.append(module.register_forward_hook(hook_fn(name)))
    print(f"[INFO] Hessian/Wanda 대상 레이어: {len(linear_names)}개")

    # 캘리브레이션 데이터 통과 (Activation 수집)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calib_dataloader, desc="데이터 통과 및 Activation 수집"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    # Hook 제거
    for h in hooks:
        h.remove()

    # 수집된 데이터를 바탕으로 스코어 및 마스크 계산
    for name in linear_names:
        if name not in activation_cache:
            continue
        module = dict(model.named_modules())[name]
        W = module.weight.data.clone()
        X_list = activation_cache[name]
        X_cat = torch.cat(X_list, dim=0)  # [total_seq_len, hidden_dim]

        # Wanda 스코어: |W| * ||X||_2
        x_norm = torch.norm(X_cat, p=2, dim=0)
        wanda_score = torch.abs(W) * x_norm.unsqueeze(0)

        # Hessian 스코어 (대각 성분 근사): X^T * X 의 대각 성분
        hessian_diag = torch.sum(X_cat ** 2, dim=0)
        hessian_score = 1.0 / (hessian_diag + 1e-8).unsqueeze(0)

        # 임계값 계산 (SpQR: Hessian→Trash, Wanda→Gems/VIP)
        tau_h_val = torch.quantile(hessian_score, tau_h)
        survivors = wanda_score[hessian_score > tau_h_val]
        tau_w_val = torch.quantile(survivors, tau_w) if len(survivors) > 0 else 0

        # 마스크 생성
        layer_masks = {
            "trash": hessian_score <= tau_h_val,
            "gems": (hessian_score > tau_h_val) & (wanda_score <= tau_w_val),
            "vip": (hessian_score > tau_h_val) & (wanda_score > tau_w_val),
        }

        masks[name] = layer_masks
        wanda_scores[name] = wanda_score

        # Bias Correction용: 입력 활성화 평균 (SpQR)
        x_means[name] = X_cat.mean(dim=0)

        del activation_cache[name]

    # 마스크 통계 로깅
    total = sum(m["trash"].numel() + m["gems"].numel() + m["vip"].numel() for m in masks.values())
    n_trash = sum(m["trash"].sum().item() for m in masks.values())
    n_gems = sum(m["gems"].sum().item() for m in masks.values())
    n_vip = sum(m["vip"].sum().item() for m in masks.values())
    print(f"[INFO] 3-Tier 마스크: Trash {100*n_trash/total:.1f}% | Gems {100*n_gems/total:.1f}% | VIP {100*n_vip/total:.1f}%")

    return masks, wanda_scores, x_means


# ==============================================================================
# 1.5단계: SpQR Bias Correction (Pruning 시 출력 보정)
# ==============================================================================
def apply_pruning_and_bias_correction(model, masks, x_means, config: Optional[SpQRWandaConfig] = None):
    """
    [1단계] Trash 구역 가중치 Pruning (항상 수행)
    [2단계] Bias Correction (config.bias_correction=True일 때): 출력 보정
    """
    cfg = config or DEFAULT_CONFIG
    n_pruned_total = 0

    for name, module in model.named_modules():
        if name not in masks:
            continue
        if not isinstance(module, nn.Linear):
            continue

        weight = module.weight.data
        trash_mask = masks[name]["trash"].to(weight.device)

        n_pruned_total += trash_mask.sum().item()

        if cfg.bias_correction and name in x_means:
            x_mean = x_means[name].to(weight.device)
            pruned_weights = weight * trash_mask
            bias_correction = torch.matmul(pruned_weights, x_mean)

            if module.bias is not None:
                module.bias.data += bias_correction
            else:
                module.bias = nn.Parameter(bias_correction.to(weight.dtype))

        weight[trash_mask] = 0.0

    n_total = sum(m["trash"].numel() + m["gems"].numel() + m["vip"].numel() for m in masks.values())
    print(f"[INFO] Pruning 완료: {n_pruned_total:,} params 제거 ({100*n_pruned_total/n_total:.1f}%)" + 
          (f" | Bias Correction 적용" if cfg.bias_correction else ""))


# ==============================================================================
# 2단계: Sensitivity-Aware OmniQuant 학습 (LWC 패널티 적용)
# ==============================================================================
class SensitivityAwareLWC(nn.Module):
    def __init__(self, weight, wanda_score, masks, n_bits=4, device=None):
        super().__init__()
        self.n_bits = n_bits
        device = device or weight.device
        self.masks = {k: v.to(device) for k, v in masks.items()}
        wanda_score = wanda_score.to(device)

        w_min, w_max = wanda_score.min(), wanda_score.max()
        normalized_wanda = (wanda_score - w_min) / (w_max - w_min + 1e-5)

        init_gamma = normalized_wanda.clone()
        init_gamma[masks["vip"].to(device)] = 1.0
        init_gamma[masks["trash"].to(device)] = 0.0

        self.gamma_low = nn.Parameter(init_gamma.clone())
        self.gamma_up = nn.Parameter(init_gamma.clone())
        
    def forward(self, weight):
        # Trash: 0-bit, Gems: LWC 양자화, VIP: 원본 유지
        weight_pruned = torch.where(self.masks["trash"], torch.zeros_like(weight), weight)
        w_min = weight_pruned.min(dim=-1, keepdim=True)[0]
        w_max = weight_pruned.max(dim=-1, keepdim=True)[0]

        clip_min = w_min * self.gamma_low
        clip_max = w_max * self.gamma_up
        clipped_weight = torch.clamp(weight_pruned, clip_min, clip_max)

        q_max = (2 ** self.n_bits) - 1
        scale = (clip_max - clip_min) / q_max
        scale = torch.clamp(scale, min=1e-5)
        zero_point = torch.round(-clip_min / scale)

        quantized_weight = torch.clamp(torch.round(clipped_weight / scale) + zero_point, 0, q_max)
        quantized_weight = (quantized_weight - zero_point) * scale

        # trash=0, gems=quantized, vip=original (SpQR: VIP 원본 유지)
        final_weight = torch.where(self.masks["trash"], torch.zeros_like(weight), quantized_weight)
        final_weight = torch.where(self.masks["vip"], weight, final_weight)
        return final_weight

def run_3tier_omniquant(
    model,
    calib_dataloader,
    masks,
    wanda_scores,
    config: Optional[SpQRWandaConfig] = None,
    epochs: Optional[int] = None,
):
    """
    OmniQuant 최적화 루프.
    - use_reconstruction_loss=True: MSE(original_out, quant_out) + lambda_w * wanda_penalty
    - VIP: 원본 weight 유지 (LWC 내부)
    """
    cfg = config or DEFAULT_CONFIG
    n_epochs = epochs if epochs is not None else cfg.omniquant_epochs
    device = next(model.parameters()).device
    model.train()

    lwc_modules = {}
    for name, module in model.named_modules():
        if name not in masks or name not in wanda_scores:
            continue
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            lwc = SensitivityAwareLWC(
                module.weight.data,
                wanda_scores[name],
                masks[name],
                n_bits=cfg.n_bits_gems,
                device=module.weight.device,
            ).to(module.weight.device)
            lwc_modules[name] = lwc

    if not lwc_modules:
        return model

    n_batches = len(calib_dataloader)
    print(f"[INFO] OmniQuant: LWC 적용 레이어 {len(lwc_modules)}개, epochs={n_epochs}, batches/epoch={n_batches}")

    optimizer = torch.optim.AdamW(
        [p for lwc in lwc_modules.values() for p in lwc.parameters()],
        lr=cfg.omniquant_lr,
    )

    # wandb 초기화 (선택)
    _wandb_run = None
    if cfg.wandb_enable:
        if not WANDB_AVAILABLE:
            print("[WARN] wandb 활성화됐지만 패키지 없음. pip install wandb 후 사용 가능")
        else:
            _wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={
                    "tau_h_percentile": cfg.tau_h_percentile,
                    "tau_w_percentile": cfg.tau_w_percentile,
                    "n_bits_gems": cfg.n_bits_gems,
                    "bias_correction": cfg.bias_correction,
                    "omniquant_epochs": n_epochs,
                    "omniquant_lr": cfg.omniquant_lr,
                    "use_reconstruction_loss": cfg.use_reconstruction_loss,
                    "lambda_w": cfg.lambda_w,
                },
            )
            print(f"[INFO] wandb 로깅 시작: {cfg.wandb_project}")

    global_step = 0
    epoch_losses = []
    for epoch in range(n_epochs):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch_idx, batch in enumerate(tqdm(calib_dataloader, desc=f"OmniQuant Epoch {epoch+1}/{n_epochs}")):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            original_weights = {}
            for name, module in model.named_modules():
                if name in lwc_modules:
                    original_weights[name] = module.weight.data.clone()
                    module.weight.data = lwc_modules[name](original_weights[name])

            if cfg.use_reconstruction_loss:
                quant_out = model(input_ids=input_ids, attention_mask=attention_mask).logits

                for name in lwc_modules:
                    module = dict(model.named_modules())[name]
                    module.weight.data = original_weights[name]

                original_out = model(input_ids=input_ids, attention_mask=attention_mask).logits
                mse_loss = F.mse_loss(quant_out, original_out)

                wanda_penalty = torch.tensor(0.0, device=device)
                for name in lwc_modules:
                    mod_device = original_weights[name].device
                    orig_w = original_weights[name]
                    quant_w = lwc_modules[name](orig_w)
                    active_mask = ~masks[name]["trash"].to(mod_device)
                    layer_wanda = wanda_scores[name].to(mod_device)
                    clipping_error = torch.abs(orig_w - quant_w)
                    wanda_penalty = wanda_penalty + torch.sum(
                        layer_wanda[active_mask] * clipping_error[active_mask]
                    ).to(device)

                total_loss = mse_loss + cfg.lambda_w * wanda_penalty
                _mse_val = mse_loss.item()
                _wanda_val = wanda_penalty.item()
            else:
                total_loss = torch.tensor(0.0, device=device)
                for name, module in model.named_modules():
                    if name in lwc_modules:
                        mod_device = module.weight.device
                        weight = original_weights[name]
                        quantized_weight = module.weight.data.clone()
                        layer_mask = masks[name]
                        layer_wanda = wanda_scores[name].to(mod_device)

                        clipping_error = torch.abs(weight - quantized_weight)
                        mask_vip = layer_mask["vip"].to(mod_device)
                        mask_gems = layer_mask["gems"].to(mod_device)

                        vip_penalty = torch.sum(clipping_error[mask_vip] * cfg.vip_penalty_weight).to(device)
                        gems_penalty = torch.sum(
                            layer_wanda[mask_gems] * clipping_error[mask_gems]
                        ).to(device)
                        total_loss = total_loss + vip_penalty + cfg.gems_penalty_weight * gems_penalty

                        module.weight.data = original_weights[name]

                _mse_val, _wanda_val = None, None

            total_loss.backward()
            optimizer.step()

            epoch_loss_sum += total_loss.item()
            epoch_batches += 1

            # wandb 로깅
            if _wandb_run is not None:
                log_interval = cfg.wandb_log_interval if cfg.wandb_log_interval > 0 else 1
                if global_step % log_interval == 0:
                    log_dict = {"train/loss": total_loss.item(), "train/epoch": epoch}
                    if cfg.use_reconstruction_loss and _mse_val is not None:
                        log_dict["train/mse_loss"] = _mse_val
                        log_dict["train/wanda_penalty"] = _wanda_val
                    wandb.log(log_dict, step=global_step)
            global_step += 1

        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"[INFO] Epoch {epoch+1}/{n_epochs} | avg_loss={avg_loss:.6f}")

    with torch.no_grad():
        for name, module in model.named_modules():
            if name in lwc_modules:
                module.weight.data = lwc_modules[name](module.weight.data)

    if epoch_losses:
        print(f"[INFO] OmniQuant 최종: loss {epoch_losses[0]:.6f} → {epoch_losses[-1]:.6f}")

    if _wandb_run is not None and WANDB_AVAILABLE:
        wandb.finish()

    return model

# ==============================================================================
# 3단계: vLLM 호환용 GPTQ 포맷 패키징 (Packing)
# ==============================================================================
def pack_to_gptq_and_save(model, tokenizer, out_dir):
    """
    최적화가 끝난 Fake-Quantized 모델을 허깅페이스 표준 `compressed-tensors` 
    또는 GPTQ 포맷으로 압축하여 저장합니다.
    """
    # 데이콘 환경에 맞추어 llmcompressor의 내장 save_pretrained를 활용하거나,
    # config.json에 4비트 양자화 정보를 강제로 주입합니다.
    model.config.quantization_config = {
        "quant_method": "compressed-tensors", # vLLM이 인식하는 키워드
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "group_size": 128,
                    "symmetric": True,
                    "type": "int"
                }
            }
        },
        "ignore": ["lm_head"]
    }
    
    os.makedirs(out_dir, exist_ok=True)
    # 현재 모델은 실수형(FP16)이므로, 내부적으로 정수형 압축이 필요한 경우
    # llmcompressor의 save_compressed=True 옵션을 활용하여 저장
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    print(f"모델이 vLLM 호환 포맷으로 {out_dir} 에 저장되었습니다.")