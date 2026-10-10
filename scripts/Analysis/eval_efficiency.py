import argparse
import importlib
import importlib.util
import json
import os
import pickle
import random
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# eval_*.py 里写的是 from dataset import ...，和它们平时被 python other_method/xxx.py
# 调用时一样，把 other_method 放进 sys.path 就能解析
sys.path.insert(0, os.path.join(ROOT, 'other_method'))

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMBD_CKPT = ('ckpt/SPO_score_Llama-3.2-1B-Instruct_ref_None_ai_detection_500_polish.raw_data'
             '_lr0.0001_bs1_beta0.05_r8_e5')
DDL_CKPT = ('ckpt/DDL_score_Llama-3.2-1B-Instruct_ref_None_ai_detection_500_polish.raw_data'
            '_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e5')
BASELINES = {'Likelihood': 'likelihood', 'LogRank': 'logrank', 'Entropy': 'entropy'}
ROBERTA_CKPT = {'RoBERTa-Base': 'ckpt/roberta_base_ai_detector/last',
                'RoBERTa-Large': 'ckpt/roberta_large_ai_detector/last'}

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--eval_data_path', type=str, default='datasets/Main/HART.json')
parser.add_argument('--eval_data_format', type=str, default='ImBD')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='默认 1：逐样本计时，测的是单样本延迟。调大相当于测吞吐，'
                         '能批处理的方法会显得更快，跨方法不再可比。')
parser.add_argument('--num_docs', type=int, default=1000, help='参与计时的样本数。')
parser.add_argument('--tokens_per_doc', type=int, default=100,
                    help='每条样本精确截到这么多 token（按 policy tokenizer 计），'
                         '脚本会重新编码核验。必须小于各方法内部的 max_token_observed。')
parser.add_argument('--max_token_observed', type=int, default=512,
                    help='各方法内部的截断长度，仅用于校验 tokens_per_doc 没设过大。')
parser.add_argument('--warmup', type=int, default=20,
                    help='计时前空跑的样本数，耗时不计入结果，用来排除首次前向的一次性开销。')
parser.add_argument('--policy_name_or_path', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
parser.add_argument('--reference_name_or_path', type=str, default='meta-llama/Llama-3.2-1B')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_path', type=str, default='./results/Analysis/Efficiency')
parser.add_argument('--save_file', type=str, default='efficiency.csv')

CSV_FIELDS = ['method', 'n_samples', 'tokens_per_sample', 'batch_size', 'n_tokens', 'seconds',
              'ms_per_sample', 'ms_per_sample_std', 'ms_per_sample_median', 's_per_1k_tokens',
              'load_seconds', 'peak_mem_GB', 'n_gpus']
# 这两个方法内部断言 batch=1，调大 batch 对它们不生效，结果里提示一下避免误读
NATIVE_BATCH1 = {'Lastde++', 'BiScope'}


def load_module(rel_path):
    """按文件路径 import 一个脚本，顺便把它所在目录放进 sys.path。"""
    path = os.path.join(ROOT, rel_path)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_scorer(args):
    """返回 score(texts) -> list[float]，模型在这里加载，不计时。"""
    method, inst, base = args.method, args.policy_name_or_path, args.reference_name_or_path

    if method in BASELINES:
        mod = load_module('other_method/eval_baselines.py')
        return mod.Baselines(scoring_model_name=inst,
                             method=mod.methods[BASELINES[method]]).to(DEV).eval()

    if method == 'LRR':
        mod = load_module('other_method/eval_lrr.py')
        return mod.LRR(scoring_model_name=inst).to(DEV).eval()

    if method == 'Fast-DetectGPT':
        mod = load_module('other_method/eval_fast_det_gpt.py')
        return mod.FastDetectGPT(scoring_model_name=inst, reference_model_name=base,
                                 discrepancy_analytic=True)

    if method == 'Binoculars':
        mod = load_module('other_method/eval_binoculars.py')
        return mod.Binoculars(observer_name_or_path=base, performer_name_or_path=inst)

    if method == 'IRM':
        mod = load_module('other_method/eval_IRM.py')
        return mod.IRMDetector(scoring_model_name=inst, reference_model_name=base)

    if method == 'DNA-DetectLLM':
        return importlib.import_module('dna_detectllm').DetectLLM(
            observer_name_or_path=base, performer_name_or_path=inst)

    if method == 'Pref-DetectGPT':
        sys.path.insert(0, os.path.join(ROOT, 'Pref'))
        return importlib.import_module('core').Pref(
            reference_name_or_path=base, policy_name_or_path=inst, sample_name_or_path=base)

    if method in ROBERTA_CKPT:
        mod = load_module('other_method/eval_roberta.py')
        ckpt = os.path.join(ROOT, ROBERTA_CKPT[method])
        return mod.RobertaDetector(scoring_model_name=ckpt).to(DEV).eval()

    if method in ('ImBD', 'DetectAnyLLM'):
        mod = load_module('other_method/core_imbd/model.py')
        ckpt, train_method = ((IMBD_CKPT, 'SPO') if method == 'ImBD' else (DDL_CKPT, 'DDL'))
        det = mod.DiscrepancyEstimator(train_method=train_method,
                                       pretrained_ckpt=os.path.join(ROOT, ckpt)).to(DEV).eval()

        def score(texts):
            tok = det.scoring_tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                                        max_length=512, return_token_type_ids=False).to(DEV)
            with torch.no_grad():
                out = det.get_discrepancy_of_scoring_and_reference_models(
                    tok['input_ids'], tok['attention_mask'])
            return out['scoring_discrepancy'].tolist()
        return score

    if method == 'Lastde++':
        mod = load_module('other_method/Lastde-Test/lastde_doubleplus.py')
        mod.model_fullnames.setdefault(inst, inst)
        mod.model_fullnames.setdefault(base, base)
        s_tok, s_model = mod.load_tokenizer(inst), mod.load_model(inst).eval()
        r_tok, r_model = mod.load_tokenizer(base), mod.load_model(base).eval()
        opt = SimpleNamespace(n_samples=100, embed_size=4, epsilon=8, tau_prime=15, max_length=512,
                              scoring_model_name=inst, reference_model_name=base)
        # Lastde++ 内部 assert batch 为 1
        return lambda texts: [mod._compute_lastde_score(t, s_tok, s_model, r_tok, r_model, opt)
                              for t in texts]

    if method == 'BiScope':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mod = load_module('other_method/biscope_utils.py')
        name = mod.MODEL_ZOO['Llama-3.2-1B-Instruct']
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map='auto').eval()
        tok = AutoTokenizer.from_pretrained(name, padding_side='left')
        tok.pad_token = tok.eos_token
        with open(os.path.join(ROOT, 'ckpt/biscope_classifier.pkl'), 'rb') as f:
            clf = pickle.load(f)
        opt = SimpleNamespace(summary_model='none', sample_clip=512,
                              detect_model='Llama-3.2-1B-Instruct')
        # BiScope 也是逐条提特征，最后过一下随机森林
        def score(texts):
            feats = [mod.detect_single_sample(opt, model, tok, None, None, t, device=DEV)
                     for t in texts]
            return clf.predict_proba(np.array(feats))[:, 1].tolist()
        return score

    raise ValueError(f'未知方法 {method}')


def exact_token_text(tok, ids, n):
    """把 token 序列截到 n 个，decode 成文本，并核验重新编码后仍然是 n 个 token。

    BPE 在 decode→encode 之间不严格可逆（空格归并、特殊字符重组等），直接 decode 出来
    的文本重新编码可能变成 n±1。这里按重新编码的结果再截一次，最多修正两轮；仍对不上
    的直接返回 None 丢弃，保证参与计时的样本长度精确一致。
    """
    text = tok.decode(ids[:n])
    for _ in range(2):
        length = len(tok(text, add_special_tokens=False).input_ids)
        if length == n:
            return text
        if length < n:
            return None
        text = tok.decode(tok(text, add_special_tokens=False).input_ids[:n])
    return None


def build_docs(args):
    """随机抽 num_docs 条长文本，每条精确截到 tokens_per_doc 个 token。"""
    from transformers import AutoTokenizer

    path = args.eval_data_path if os.path.isabs(args.eval_data_path) \
        else os.path.join(ROOT, args.eval_data_path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if args.eval_data_format == 'MIRAGE':
        pairs = [(d['human'], d['llm']) for d in raw]
    else:
        pairs = list(zip(raw['human'], raw['llm']))
    # human / llm 都要打分，两列都算进候选池
    candidates = [t for pair in pairs for t in pair]

    tok = AutoTokenizer.from_pretrained(args.policy_name_or_path)
    n = args.tokens_per_doc
    if n >= args.max_token_observed:
        raise SystemExit(f'--tokens_per_doc={n} 不小于方法内部截断 {args.max_token_observed}，'
                         f'会被二次截断。')

    ids = tok(candidates, add_special_tokens=False).input_ids
    pool = [i for i, seq in enumerate(ids) if len(seq) >= n]
    # 多抽一些做备用：核验这一关会丢掉极少数 decode→encode 对不齐的样本
    rng = random.Random(args.seed)
    order = rng.sample(pool, len(pool))

    docs, dropped = [], 0
    for i in order:
        if len(docs) == args.num_docs:
            break
        text = exact_token_text(tok, ids[i], n)
        if text is None:
            dropped += 1
            continue
        docs.append(text)

    if not docs:
        return [], 0
    lens = [len(s) for s in tok(docs, add_special_tokens=False).input_ids]
    assert set(lens) == {n}, f'截断核验失败：出现了 {sorted(set(lens))} 这些长度'
    print(f'{len(docs)} 条样本，每条精确 {n} token（候选池 {len(pool)} 条，'
          f'decode/encode 对不齐丢弃 {dropped} 条）')
    if len(docs) < args.num_docs:
        print(f'[warn] 不足 {args.num_docs} 条，实际用 {len(docs)} 条。')
    return docs, len(docs) * n


def count_method_tokens(args, docs):
    """RoBERTa 用自己的 BPE，记一下它实际看到多少 token，方便解释表里的差异。"""
    if args.method not in ROBERTA_CKPT:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, ROBERTA_CKPT[args.method]))
        return sum(len(s) for s in tok(docs, add_special_tokens=True).input_ids)
    except Exception as e:
        print(f'[warn] 方法自身 token 数统计跳过：{e}')
        return None


def sync_all():
    """逐卡同步。不带参数的 synchronize 只同步当前设备，多卡方法会漏掉另一张卡。"""
    if DEV != 'cuda':
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def peak_memory_gb():
    if DEV != 'cuda':
        return 0.0
    return sum(torch.cuda.max_memory_allocated(i)
               for i in range(torch.cuda.device_count())) / 1024 ** 3


def upsert_csv(path, row):
    """同一个方法重复测只保留最后一次，避免 append 出一堆重复行。"""
    rows = {}
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        header = lines[0].split(',')
        for line in lines[1:]:
            vals = line.split(',')
            if header == CSV_FIELDS and len(vals) == len(CSV_FIELDS):
                rows[vals[0]] = vals
    rows[row[0]] = row
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(CSV_FIELDS) + '\n')
        for vals in rows.values():
            f.write(','.join(str(v) for v in vals) + '\n')


def main():
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    docs, n_tokens = build_docs(args)
    if not docs:
        raise SystemExit(f'没有长度 >= {args.tokens_per_doc} token 的文本，请调小 --tokens_per_doc。')
    n_docs = len(docs)
    method_tokens = count_method_tokens(args, docs)

    batch_size = args.eval_batch_size
    if args.method in NATIVE_BATCH1 and batch_size > 1:
        print(f'[info] {args.method} 内部只能逐条推理，--eval_batch_size={batch_size} 不会生效。')

    t0 = time.perf_counter()
    scorer = build_scorer(args)
    sync_all()
    load_seconds = time.perf_counter() - t0

    score = scorer.compute_score if hasattr(scorer, 'compute_score') else scorer

    # 预热：首次前向含 CUDA context 初始化、kernel 选择、显存分配、GPU 升频，
    # 这段耗时和返回值全部丢弃，计时从下面的循环才开始。
    # 用和正式计时相同的 batch 形状预热，否则 kernel 还要为新形状重选一次
    for i in range(0, args.warmup, batch_size):
        start = i % n_docs
        score(docs[start:start + batch_size] or docs[:batch_size])
    sync_all()

    if DEV == 'cuda':
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    # 计时：默认 batch=1 就是逐样本，每次调用前后都做全卡同步
    batches = [docs[i:i + batch_size] for i in range(0, n_docs, batch_size)]
    latencies = []
    for batch in batches:
        sync_all()
        t = time.perf_counter()
        score(batch)
        sync_all()
        latencies.append(time.perf_counter() - t)

    elapsed = sum(latencies)
    # batch>1 时把一次调用的耗时摊到它包含的样本上；batch=1 时就是原始的单样本耗时
    ms = [lat / len(b) * 1000 for lat, b in zip(latencies, batches)]
    ms_mean = statistics.mean(ms)
    ms_std = statistics.stdev(ms) if len(ms) > 1 else 0.0
    ms_median = statistics.median(ms)
    s_per_1k = elapsed / n_tokens * 1000
    peak_mem = peak_memory_gb()
    n_gpus = torch.cuda.device_count() if DEV == 'cuda' else 0

    print(f'{args.method}: {n_docs} 条 x {args.tokens_per_doc} token | bs={batch_size} | '
          f'{ms_mean:.2f}±{ms_std:.2f} ms/sample（中位数 {ms_median:.2f}）'
          f' => {s_per_1k:.4f} s/1k tokens | 总计 {elapsed:.2f} s | '
          f'load {load_seconds:.1f} s | peak {peak_mem:.2f} GB | {n_gpus} GPU')

    row = [args.method, n_docs, args.tokens_per_doc, batch_size, n_tokens, f'{elapsed:.3f}',
           f'{ms_mean:.3f}', f'{ms_std:.3f}', f'{ms_median:.3f}', f'{s_per_1k:.4f}',
           f'{load_seconds:.2f}', f'{peak_mem:.2f}', n_gpus]

    os.makedirs(args.save_path, exist_ok=True)
    upsert_csv(os.path.join(args.save_path, args.save_file), row)

    # 逐样本的原始耗时留一份，方便事后复查分布 / 重画图
    raw_dir = os.path.join(args.save_path, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, f'{args.method}.json'), 'w', encoding='utf-8') as f:
        json.dump({'config': vars(args), 'n_samples': n_docs, 'n_tokens': n_tokens,
                   'method_own_n_tokens': method_tokens, 'n_gpus': n_gpus,
                   'gpu_names': [torch.cuda.get_device_name(i) for i in range(n_gpus)],
                   'load_seconds': load_seconds, 'peak_mem_GB': peak_mem,
                   'seconds': elapsed, 'ms_per_sample': ms_mean, 'ms_per_sample_std': ms_std,
                   'ms_per_sample_median': ms_median, 's_per_1k_tokens': s_per_1k,
                   'latencies_ms': ms},
                  f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
