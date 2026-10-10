from typing import Union

import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import assert_tokenizer_consistency
import torch.nn.functional as F

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# 在当前可见设备内，保持与 eval_binoculars 相同的策略：
# observer 放在第 0 张卡，performer 放在第 1 张卡（如果只有一张卡，则都在同一张卡上）
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Pref(object):
    def __init__(self,
                 reference_name_or_path: str = "meta-llama/Llama-3.2-1B",
                 policy_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(reference_name_or_path, policy_name_or_path)

        self.reference_model = AutoModelForCausalLM.from_pretrained(reference_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   # token=huggingface_config["TOKEN"]
                                                                   )
        self.policy_model = AutoModelForCausalLM.from_pretrained(policy_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    # token=huggingface_config["TOKEN"]
                                                                    )
        self.reference_model.eval()
        self.policy_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(reference_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed


    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.reference_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        reference_logits = self.reference_model(**encodings.to(DEVICE_1)).logits
        policy_logits = self.policy_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return reference_logits, policy_logits



    def cleanup(self):
        if self.reference_model is not None:
            del self.reference_model
            del self.policy_model
            self.reference_model = None
            self.policy_model = None
        torch.cuda.empty_cache()

    def prob_compute(self, encoding: transformers.BatchEncoding,
                        logits: torch.Tensor,
                        temperature: float = 1.0):
        # 1. 准备工作：对齐序列长度 [batch, seq_len - 1, vocab]
        shifted_logits = logits[..., :-1, :] / temperature
        labels_std = encoding.input_ids[..., 1:].unsqueeze(-1)

        probs = torch.softmax(shifted_logits, dim=-1)
        lprobs = torch.log_softmax(shifted_logits, dim=-1)
        
        # 3. 提取 nppl_std: 实际文本对应的 log_prob (即 -PPL 的核心部分)
        # nce_std 维度: [batch, seq_len - 1]
        ll = lprobs.gather(dim=-1, index=labels_std).squeeze(-1)

        # 返回你需要的两个核心指标
        return ll, probs, lprobs
    
    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        reference_logits, policy_logits = self._get_logits(encodings)
        
        reference_logits = reference_logits.to(policy_logits.device)
        
        ll_p, probs_p, lprobs_p = self.prob_compute(encodings, policy_logits)
        ll_o, probs_o, lprobs_o = self.prob_compute(encodings, reference_logits)
        
        attention = encodings.attention_mask[:, 1:].float()
        
        # 1. 计算实际token的差异（每个位置）
        actual_diff = ll_p - ll_o  # [batch, seq_len]
        
        # 2. 计算期望差异（每个位置）
        mean_diff = (probs_o * lprobs_p).sum(dim=-1) - (probs_o * lprobs_o).sum(dim=-1)  # [batch, seq_len]
        
        # 3. 计算方差（用于z-score标准化）
        # Var[log P_o] under P_p distribution
        var = (probs_o * torch.square(lprobs_p-lprobs_o)).sum(dim=-1) - torch.square((probs_o * (lprobs_p-lprobs_o)).sum(dim=-1))
        
        # 4. 先应用attention mask并聚合
        actual_diff = (actual_diff * attention).sum(dim=1)  # [batch]
        mean_diff = (mean_diff * attention).sum(dim=1)  # [batch]
        var = (var * attention).sum(dim=1)  # [batch]
        
        # 5. 再计算z-score
        # Pref_scores = (actual_diff - mean_diff) / (var.sqrt() + 1e-8)  # [batch]
        
        Pref_scores = (actual_diff - mean_diff) / (var.sqrt() + 1e-8)  # [batch]

        
        
        Pref_scores = Pref_scores.tolist()
        return Pref_scores

