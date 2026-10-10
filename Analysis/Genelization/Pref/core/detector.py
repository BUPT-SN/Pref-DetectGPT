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
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1
# 为 sample 模型分配设备（如果有第 3 张卡则用 cuda:2，否则复用 DEVICE_1）
DEVICE_3 = "cuda:2" if torch.cuda.device_count() > 2 else DEVICE_1


class Pref(object):
    def __init__(self,
                 reference_name_or_path: str = "meta-llama/Llama-3.2-1B",
                 policy_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
                 sample_name_or_path: str = "meta-llama/Llama-3.2-1B",  # 新增 sample 模型路径
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
        
        # 加载 sample_model
        self.sample_model = AutoModelForCausalLM.from_pretrained(sample_name_or_path,
                                                                  device_map={"": DEVICE_3},
                                                                  trust_remote_code=True,
                                                                  torch_dtype=torch.bfloat16 if use_bfloat16
                                                                  else torch.float32,
                                                                  # token=huggingface_config["TOKEN"]
                                                                  )

        self.reference_model.eval()
        self.policy_model.eval()
        self.sample_model.eval()

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
    def _get_logits(self, encodings: transformers.BatchEncoding) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference_logits = self.reference_model(**encodings.to(DEVICE_1)).logits
        policy_logits = self.policy_model(**encodings.to(DEVICE_2)).logits
        sample_logits = self.sample_model(**encodings.to(DEVICE_3)).logits  # 获取 sample 的 logits
        
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return reference_logits, policy_logits, sample_logits


    def cleanup(self):
        if self.reference_model is not None:
            del self.reference_model
            del self.policy_model
            del self.sample_model
            self.reference_model = None
            self.policy_model = None
            self.sample_model = None
        torch.cuda.empty_cache()

    def prob_compute(self, encoding: transformers.BatchEncoding,
                        logits: torch.Tensor,
                        temperature: float = 1.0):
        # 1. 准备工作：对齐序列长度 [batch, seq_len - 1, vocab]
        shifted_logits = logits[..., :-1, :] / temperature
        
        # ====== 核心修复：确保 labels_std 动态转移到当前 logits 所在的设备 ======
        labels_std = encoding.input_ids[..., 1:].unsqueeze(-1).to(logits.device)

        probs = torch.softmax(shifted_logits, dim=-1)
        lprobs = torch.log_softmax(shifted_logits, dim=-1)
        
        # 3. 提取 nppl_std: 实际文本对应的 log_prob (即 -PPL 的核心部分)
        ll = lprobs.gather(dim=-1, index=labels_std).squeeze(-1)

        return ll, probs, lprobs
    
    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        
        # 获取三个模型的 logits
        reference_logits, policy_logits, sample_logits = self._get_logits(encodings)
        
        # 统一移动到 policy_logits 所在的设备进行后续的张量运算
        reference_logits = reference_logits.to(policy_logits.device)
        sample_logits = sample_logits.to(policy_logits.device)
        
        ll_p, probs_p, lprobs_p = self.prob_compute(encodings, policy_logits)
        ll_o, probs_o, lprobs_o = self.prob_compute(encodings, reference_logits)
        ll_s, probs_s, lprobs_s = self.prob_compute(encodings, sample_logits) # 获取 sample 分布
        
        # 确保 attention mask 也在同一个计算设备上
        attention = encodings.attention_mask[:, 1:].float().to(policy_logits.device)
        
        # 1. 计算实际token的差异（每个位置）
        actual_diff = ll_p - ll_o  # [batch, seq_len]
        
        # 2. 计算期望差异（每个位置）- 使用 probs_s 替换 probs_o
        mean_diff = (probs_s * lprobs_p).sum(dim=-1) - (probs_s * lprobs_o).sum(dim=-1)  # [batch, seq_len]
        
        # 3. 计算方差（用于z-score标准化）- 使用 probs_s 替换 probs_o
        var = (probs_s * torch.square(lprobs_p-lprobs_o)).sum(dim=-1) - torch.square((probs_s * (lprobs_p-lprobs_o)).sum(dim=-1))
        
        # 4. 先应用attention mask并聚合
        actual_diff = (actual_diff * attention).sum(dim=1)  # [batch]
        mean_diff = (mean_diff * attention).sum(dim=1)  # [batch]
        var = (var * attention).sum(dim=1)  # [batch]
        
        # 5. 再计算z-score
        Pref_scores = (actual_diff - mean_diff) / (var.sqrt() + 1e-8)  # [batch]
        
        Pref_scores = Pref_scores.tolist()
        return Pref_scores