import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch

from core import Pref
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1, TPR_at_FPR10

parser = argparse.ArgumentParser()
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD.')
parser.add_argument('--save_path', type=str, default='./results/Pref')
parser.add_argument('--save_file', type=str, default='eval_Pref.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--reference_name_or_path', type=str, default='tiiuae/falcon-7b', help='Observer model name or local path.')
parser.add_argument('--policy_name_or_path', type=str, default='tiiuae/falcon-7b-instruct', help='Performer model name or local path.')
parser.add_argument('--sample_name_or_path', type=str, default='tiiuae/falcon-7b', help='Sample model name or local path.')
parser.add_argument('--max_token_observed', type=int, default=512)

# ====== 新增：添加 prompt 命令行参数 ======
parser.add_argument('--prompt', type=str, default='', help='The prompt prefix to prepend to the text but ignore in scoring.')

torch.set_grad_enabled(False)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    detector = Pref(
        reference_name_or_path=args.reference_name_or_path,
        policy_name_or_path=args.policy_name_or_path,
        sample_name_or_path=args.sample_name_or_path, 
        max_token_observed=args.max_token_observed,
        prompt=args.prompt  # ====== 新增：将 prompt 传给 Pref 类 ======
    )

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    human_scores = []
    llm_scores = []

    iterator = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc="Evaluating Pref-DetectGPT",
    )

    for idx, item in iterator:
        # ====== 这里不需要再传 prompt，因为 Pref 内部已经保存了 ======
        h_scores = detector.compute_score(item['human'])
        l_scores = detector.compute_score(item['llm'])
        human_scores.extend([s for s in h_scores])
        llm_scores.extend([s for s in l_scores])

    fpr, tpr, eval_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
    prec, recall, eval_aupr = AUPR(neg_list=human_scores, pos_list=llm_scores)
    tpr_at_5 = TPR_at_FPR5(neg_list=human_scores, pos_list=llm_scores)
    tpr_at_10 = TPR_at_FPR10(neg_list=human_scores, pos_list=llm_scores)
    eval_f1 = F1(neg_list=human_scores, pos_list=llm_scores)

    human_score_mean = float(torch.mean(torch.tensor(human_scores)))
    human_score_std = float(torch.std(torch.tensor(human_scores)))
    llm_score_mean = float(torch.mean(torch.tensor(llm_scores)))
    llm_score_std = float(torch.std(torch.tensor(llm_scores)))

    print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f} | TPR@FPR5%: {tpr_at_5:.4f} | TPR@FPR10%: {tpr_at_10:.4f} | F1: {eval_f1:.4f}')

    result_dict = {
        'method': 'Pref',
        'reference_model_path': args.reference_name_or_path,
        'policy_model_path': args.policy_name_or_path,
        'sample_model_path': args.sample_name_or_path,
        'prompt_used': args.prompt,  # ====== 新增：在结果字典中记录使用的 prompt ======
        'eval_dataset': args.eval_data_path.split('/')[-1].split('.json')[0],
        'eval_batch_size': args.eval_batch_size,
        'human_score_mean': human_score_mean,
        'human_score_std': human_score_std,
        'llm_score_mean': llm_score_mean,
        'llm_score_std': llm_score_std,
        'AUROC': eval_auroc,
        'AUPR': eval_aupr,
        'TPR_AT_FPR_5%': tpr_at_5,
        'TPR_AT_FPR_10%': tpr_at_10,
        'F1': eval_f1,
        'human_scores': human_scores,
        'llm_scores': llm_scores
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)