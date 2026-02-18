import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

# ==============================================================================
# 1단계: Hessian 및 Wanda 스코어 계산 (3-Tier 마스크 생성)
# ==============================================================================
def calculate_hessian_wanda(model, calib_dataloader, tau_h_percentile=0.3, tau_w_percentile=0.5):
    """
    캘리브레이션 데이터를 돌려서 각 Linear 레이어의 Hessian 대각성분과 Wanda 점수를 계산합니다.
    calib_dataloader는 {"input_ids": tensor, "attention_mask": tensor} 형태의 배치를 반환해야 합니다.
    """
    masks = {}
    wanda_scores = {}
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

        # 임계값 계산
        tau_h = torch.quantile(hessian_score, tau_h_percentile)
        survivors = wanda_score[hessian_score > tau_h]
        tau_w = torch.quantile(survivors, tau_w_percentile) if len(survivors) > 0 else 0

        # 마스크 생성
        layer_masks = {
            "trash": hessian_score <= tau_h,
            "gems": (hessian_score > tau_h) & (wanda_score <= tau_w),
            "vip": (hessian_score > tau_h) & (wanda_score > tau_w),
        }

        masks[name] = layer_masks
        wanda_scores[name] = wanda_score

        del activation_cache[name]

    return masks, wanda_scores


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
        weight_pruned = torch.where(self.masks['trash'], torch.zeros_like(weight), weight)
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
        
        final_weight = torch.where(self.masks['trash'], torch.zeros_like(weight), quantized_weight)
        return final_weight

def run_3tier_omniquant(model, calib_dataloader, masks, wanda_scores, epochs=20):
    """
    OmniQuant의 최적화 루프입니다. 블록 단위로 쪼개서 최적화하는 것이 정석이나,
    해커톤 환경에 맞게 간소화된 글로벌 최적화 루프 예시입니다.
    """
    device = next(model.parameters()).device
    model.train()

    # 양자화 대상 레이어에 LWC 모듈 부착 (masks에 있는 레이어만)
    lwc_modules = {}
    for name, module in model.named_modules():
        if name not in masks or name not in wanda_scores:
            continue
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            lwc = SensitivityAwareLWC(
                module.weight.data,
                wanda_scores[name],
                masks[name],
                device=module.weight.device,
            ).to(module.weight.device)
            lwc_modules[name] = lwc

    if not lwc_modules:
        return model

    optimizer = torch.optim.AdamW(
        [p for lwc in lwc_modules.values() for p in lwc.parameters()], lr=1e-2
    )

    for epoch in range(epochs):
        for batch in tqdm(calib_dataloader, desc=f"OmniQuant Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # LWC를 통과한 가중치로 임시 교체
            original_weights = {}
            for name, module in model.named_modules():
                if name in lwc_modules:
                    original_weights[name] = module.weight.data.clone()
                    module.weight.data = lwc_modules[name](original_weights[name])

            # 패널티 계산
            total_penalty = torch.tensor(0.0, device=device)
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

                    vip_penalty = torch.sum(clipping_error[mask_vip] * 100.0).to(device)
                    gems_penalty = torch.sum(
                        layer_wanda[mask_gems] * clipping_error[mask_gems]
                    ).to(device)
                    total_penalty = total_penalty + vip_penalty + 0.01 * gems_penalty

                    module.weight.data = original_weights[name]

            total_penalty.backward()
            optimizer.step()

    # 학습 종료 후 최종 가중치 적용
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in lwc_modules:
                module.weight.data = lwc_modules[name](module.weight.data)

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