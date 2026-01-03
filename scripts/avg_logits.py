#!/usr/bin/env python3
# -*- coding: utf- -*-
"""
ensemble_logits.py
对 fold0_logits.csv … fold4_logits.csv 做 logits 平均并生成提交文件
用法：
    python ensemble_logits.py  results/2026-01-02_18-48-29/
"""
import sys, pathlib, pandas as pd, numpy as np

def main(res_dir: pathlib.Path):
    # 找到所有 fold*_logits.csv
    files = sorted(res_dir.glob('fold*_logits.csv'))
    if not files:
        raise FileNotFoundError('未找到 fold*_logits.csv')

    # 依次读取并堆叠
    logits_list = []
    for f in files:
        df = pd.read_csv(f, index_col='id')
        logits_list.append(df.values)
    avg_logits = np.mean(logits_list, axis=0)          # shape: (N_samples, 21)

    # 取最大值为预测类别
    pred = avg_logits.argmax(axis=1)

    # 生成提交文件
    sub = pd.DataFrame({'id': df.index, 'categories': pred})
    out_file = res_dir / 'submission2.csv'
    sub.to_csv(out_file, index=False)
    print(f'已生成 {out_file} ，共 {len(sub)} 条样本')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python ensemble_logits.py <result_dir>')
        sys.exit(1)
    main(pathlib.Path(sys.argv[1]).expanduser().resolve())
