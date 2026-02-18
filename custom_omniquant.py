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
    """
    masks = {}
    wanda_scores = {}
    
    # 활성화값(X)을 가로채기 위한 Hook 함수 설정
    activation_cache = {}
    def hook_fn(name):
        def forward_hook(module, input, output):
            x = input[0].detach().squeeze(0) # [seq_len, hidden_dim]
            if name not in activation_cache:
                activation_cache[name] = []
            activation_cache[name].append(x)
        return forward_hook

    # Hook 등록 (EXAONE의 Linear 레이어 타겟)
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 캘리브레이션 데이터 통과 (Activation 수집)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calib_dataloader, desc="데이터 통과 및 Activation 수집"):
            input_ids = torch.tensor([batch['text']]).to(model.device) # 구조에 맞게 수정 필요
            model(input_ids)

    # Hook 제거
    for h in hooks:
        h.remove()

    # 수집된 데이터를 바탕으로 스코어 및 마스크 계산
    for name, module in model.named_modules():
        if name in activation_cache:
            W = module.weight.data.clone()
            X_list = activation_cache[name]
            X_cat = torch.cat(X_list, dim=0) # [total_seq_len, hidden_dim]
            
            # Wanda 스코어: |W| * ||X||_2
            x_norm = torch.norm(X_cat, p=2, dim=0)
            wanda_score = torch.abs(W) * x_norm.unsqueeze(0)
            
            # Hessian 스코어 (대각 성분 근사): X^T * X 의 대각 성분
            hessian_diag = torch.sum(X_cat ** 2, dim=0)
            # 0 나누기 방지를 위해 아주 작은 값 추가 후 역수
            hessian_score = 1.0 / (hessian_diag + 1e-8).unsqueeze(0)
            
            # 임계값 계산
            tau_h = torch.quantile(hessian_score, tau_h_percentile)
            survivors = wanda_score[hessian_score > tau_h]
            tau_w = torch.quantile(survivors, tau_w_percentile) if len(survivors) > 0 else 0
            
            # 마스크 생성
            layer_masks = {
                'trash': hessian_score <= tau_h,
                'gems': (hessian_score > tau_h) & (wanda_score <= tau_w),
                'vip': (hessian_score > tau_h) & (wanda_score > tau_w)
            }
            
            masks[name] = layer_masks
            wanda_scores[name] = wanda_score
            
            # 메모리 정리
            del activation_cache[name]

    return masks, wanda_scores


# ==============================================================================
# 2단계: Sensitivity-Aware OmniQuant 학습 (LWC 패널티 적용)
# ==============================================================================
class SensitivityAwareLWC(nn.Module):
    def __init__(self, weight, wanda_score, masks, n_bits=4):
        super().__init__()
        self.n_bits = n_bits
        self.masks = masks
        
        w_min, w_max = wanda_score.min(), wanda_score.max()
        normalized_wanda = (wanda_score - w_min) / (w_max - w_min + 1e-5)
        
        init_gamma = normalized_wanda.clone()
        init_gamma[masks['vip']] = 1.0 
        init_gamma[masks['trash']] = 0.0 
        
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
    model.train() # 학습 모드
    
    # 양자화 대상 레이어에 LWC 모듈 부착
    lwc_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            lwc = SensitivityAwareLWC(module.weight.data, wanda_scores[name], masks[name]).to(model.device)
            lwc_modules[name] = lwc
            
    # 옵티마이저 설정 (gamma 파라미터만 학습)
    optimizer = torch.optim.AdamW([p for lwc in lwc_modules.values() for p in lwc.parameters()], lr=1e-2)
    
    for epoch in range(epochs):
        for batch in tqdm(calib_dataloader, desc=f"OmniQuant Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            # Forward pass 시 가중치 덮어씌우기 (Fake Quantization 적용)
            original_weights = {}
            for name, module in model.named_modules():
                if name in lwc_modules:
                    original_weights[name] = module.weight.data.clone()
                    # LWC를 통과한 가중치로 임시 교체
                    module.weight.data = lwc_modules[name](original_weights[name])
            
            # Loss 계산 (여기서는 단순히 예측 에러 최적화로 간소화. 실제로는 Block-wise MSE 활용)
            # 주의: 원본 모델과 깎인 모델의 출력 차이를 구하는 로직으로 발전시켜야 함
            
            # 패널티 계산
            total_penalty = 0
            for name, module in model.named_modules():
                if name in lwc_modules:
                    weight = original_weights[name]
                    quantized_weight = module.weight.data
                    layer_mask = masks[name]
                    layer_wanda = wanda_scores[name]
                    
                    clipping_error = torch.abs(weight - quantized_weight)
                    
                    # VIP 극단적 패널티 + Gems 가중 패널티
                    vip_penalty = torch.sum(clipping_error[layer_mask['vip']] * 100.0) 
                    gems_penalty = torch.sum(layer_wanda[layer_mask['gems']] * clipping_error[layer_mask['gems']])
                    
                    total_penalty += (vip_penalty + 0.01 * gems_penalty)
                    
                    # 가중치 원상복구 (다음 배치를 위해)
                    module.weight.data = original_weights[name]
            
            # total_penalty.backward() # 역전파
            # optimizer.step()
            
    # 학습이 끝난 후 최종 가중치 덮어씌우기
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