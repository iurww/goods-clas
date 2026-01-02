import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import json
import os

def save_confusion_matrix(path, predictions, labels, num_classes):
    
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[f'C{i}' for i in range(num_classes)])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=True)

    plt.xticks(rotation=45, ha='right')
    plt.title('21-Class Confusion Matrix')
    plt.tight_layout()
    # plt.show()
    plt.savefig(path, dpi=150)
    
    

def plot_confusion_matrix(all_labels, all_preds, num_classes, save_dir, fold_idx=0):

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    data_path = os.path.join(save_dir, f'confusion_matrix_fold{fold_idx}_data.json')
    with open(data_path, 'w') as f:
        json.dump({
            'confusion_matrix': cm.tolist(),
            'labels': list(range(num_classes)),
            'true_labels': all_labels.tolist(),
            'pred_labels': all_preds.tolist()
        }, f, indent=2)
    
    # 1. 绘制原始混淆矩阵（未归一化）
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'{i:02d}' for i in range(num_classes)],
                yticklabels=[f'{i:02d}' for i in range(num_classes)],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix (Raw Counts) - Fold {fold_idx}', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    img_path_raw = os.path.join(save_dir, f'confusion_matrix_fold{fold_idx}_raw.png')
    plt.savefig(img_path_raw, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制归一化混淆矩阵（按行归一化，显示召回率）
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=[f'{i:02d}' for i in range(num_classes)],
                yticklabels=[f'{i:02d}' for i in range(num_classes)],
                cbar_kws={'label': 'Recall'})
    
    plt.title(f'Confusion Matrix (Normalized) - Fold {fold_idx}', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    img_path_normalized = os.path.join(save_dir, f'confusion_matrix_fold{fold_idx}_normalized.png')
    plt.savefig(img_path_normalized, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(f"Confusion matrix (raw) saved to: {img_path_raw}")
    # print(f"Confusion matrix (normalized) saved to: {img_path_normalized}")
    # print(f"Confusion matrix data saved to: {data_path}")
    
    return cm, img_path_raw, img_path_normalized, data_path


def plot_roc_curves(all_labels, all_probs, num_classes, save_dir, fold_idx=0):
    
    # 将标签二值化
    y_true_bin = label_binarize(all_labels, classes=range(num_classes))
    
    # 存储每个类别的ROC数据
    roc_data = {}
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    # 计算每个类别的ROC曲线
    for i in range(num_classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        roc_auc_dict[i] = roc_auc
        
        roc_data[f'class_{i:02d}'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }
    
    # 计算micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    roc_data['micro_average'] = {
        'fpr': fpr_micro.tolist(),
        'tpr': tpr_micro.tolist(),
        'auc': float(roc_auc_micro)
    }
    
    # 计算macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= num_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)
    roc_data['macro_average'] = {
        'fpr': all_fpr.tolist(),
        'tpr': mean_tpr.tolist(),
        'auc': float(roc_auc_macro)
    }
    
    # 保存ROC数据
    data_path = os.path.join(save_dir, f'roc_curves_fold{fold_idx}_data.json')
    with open(data_path, 'w') as f:
        json.dump(roc_data, f, indent=2)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
    
    # 绘制micro-average和macro-average
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-avg (AUC = {roc_auc_micro:.3f})', 
             linewidth=2, linestyle=':', color='deeppink')
    plt.plot(all_fpr, mean_tpr, label=f'Macro-avg (AUC = {roc_auc_macro:.3f})', 
             linewidth=2, linestyle=':', color='navy')
    
    # 选择性绘制部分类别（避免图例过多）
    if num_classes <= 10:
        # 类别少时，绘制所有类别
        for i in range(num_classes):
            plt.plot(fpr_dict[i], tpr_dict[i], lw=1, alpha=0.7,
                    label=f'Class {i:02d} (AUC = {roc_auc_dict[i]:.3f})')
    else:
        # 类别多时，只绘制AUC最高和最低的几个类别
        sorted_classes = sorted(roc_auc_dict.items(), key=lambda x: x[1], reverse=True)
        top_classes = [c[0] for c in sorted_classes[:3]]
        bottom_classes = [c[0] for c in sorted_classes[-3:]]
        selected_classes = top_classes + bottom_classes
        
        for i in selected_classes:
            label_type = 'Top' if i in top_classes else 'Bottom'
            plt.plot(fpr_dict[i], tpr_dict[i], lw=1, alpha=0.7,
                    label=f'{label_type} - Class {i:02d} (AUC = {roc_auc_dict[i]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Fold {fold_idx}', fontsize=14, pad=20)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    img_path = os.path.join(save_dir, f'roc_curves_fold{fold_idx}.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(f"ROC curves saved to: {img_path}")
    # print(f"ROC curves data saved to: {data_path}")
    # print(f"Micro-average AUC: {roc_auc_micro:.4f}")
    # print(f"Macro-average AUC: {roc_auc_macro:.4f}")
    
    return roc_data, img_path, data_path


def plot_pr_curves(all_labels, all_probs, num_classes, save_dir, fold_idx=0):

    # 将标签二值化
    y_true_bin = label_binarize(all_labels, classes=range(num_classes))
    
    # 存储每个类别的PR数据
    pr_data = {}
    precision_dict = {}
    recall_dict = {}
    precision_interp_dict = {}
    recall_interp_dict = {}
    ap_dict = {}
    
    # 计算每个类别的PR曲线
    for i in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], all_probs[:, i])
        
        # 保存原始数据
        precision_dict[i] = precision
        recall_dict[i] = recall
        ap_dict[i] = ap
        
        # 插值平滑：在标准recall点上插值，并确保单调递减
        recall_interp = np.linspace(0, 1, 100)
        precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])
        # 确保precision单调递减（从右到左取最大值）
        precision_interp = np.maximum.accumulate(precision_interp[::-1])[::-1]
        
        precision_interp_dict[i] = precision_interp
        recall_interp_dict[i] = recall_interp
        
        pr_data[f'class_{i:02d}'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'precision_interpolated': precision_interp.tolist(),
            'recall_interpolated': recall_interp.tolist(),
            'thresholds': thresholds.tolist() if len(thresholds) > 0 else [],
            'average_precision': float(ap)
        }
    
    # 计算micro-average PR curve
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), all_probs.ravel())
    ap_micro = average_precision_score(y_true_bin, all_probs, average='micro')
    
    # 插值平滑micro-average
    recall_micro_interp = np.linspace(0, 1, 100)
    precision_micro_interp = np.interp(recall_micro_interp, recall_micro[::-1], precision_micro[::-1])
    precision_micro_interp = np.maximum.accumulate(precision_micro_interp[::-1])[::-1]
    
    pr_data['micro_average'] = {
        'precision': precision_micro.tolist(),
        'recall': recall_micro.tolist(),
        'precision_interpolated': precision_micro_interp.tolist(),
        'recall_interpolated': recall_micro_interp.tolist(),
        'average_precision': float(ap_micro)
    }
    
    # 计算macro-average AP
    ap_macro = np.mean(list(ap_dict.values()))
    pr_data['macro_average'] = {
        'average_precision': float(ap_macro)
    }
    
    # 保存PR数据
    data_path = os.path.join(save_dir, f'pr_curves_fold{fold_idx}_data.json')
    with open(data_path, 'w') as f:
        json.dump(pr_data, f, indent=2)
    
    # 绘制PR曲线（使用平滑后的曲线）
    plt.figure(figsize=(10, 8))
    
    # 绘制micro-average (使用平滑曲线)
    plt.plot(recall_micro_interp, precision_micro_interp,
             label=f'Micro-avg (AP = {ap_micro:.3f})',
             linewidth=2.5, linestyle='-', color='deeppink')
    
    # 绘制基线（random classifier）
    baseline = np.sum(y_true_bin) / y_true_bin.size
    plt.axhline(y=baseline, color='k', linestyle='--', lw=2, 
                label=f'Random (AP ≈ {baseline:.3f})')
    
    # 选择性绘制部分类别
    if num_classes <= 10:
        # 类别少时，绘制所有类别
        for i in range(num_classes):
            plt.plot(recall_interp_dict[i], precision_interp_dict[i],
                    lw=1.5, alpha=0.7,
                    label=f'Class {i:02d} (AP = {ap_dict[i]:.3f})')
    else:
        # 类别多时，只绘制AP最高和最低的几个类别
        sorted_classes = sorted(ap_dict.items(), key=lambda x: x[1], reverse=True)
        top_classes = [c[0] for c in sorted_classes[:3]]
        bottom_classes = [c[0] for c in sorted_classes[-3:]]
        selected_classes = top_classes + bottom_classes
        
        for i in selected_classes:
            label_type = 'Top' if i in top_classes else 'Bottom'
            plt.plot(recall_interp_dict[i], precision_interp_dict[i],
                    lw=1.5, alpha=0.7,
                    label=f'{label_type} - Class {i:02d} (AP = {ap_dict[i]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - Fold {fold_idx}', fontsize=14, pad=20)
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    img_path = os.path.join(save_dir, f'pr_curves_fold{fold_idx}.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(f"PR curves saved to: {img_path}")
    # print(f"PR curves data saved to: {data_path}")
    # print(f"Micro-average AP: {ap_micro:.4f}")
    # print(f"Macro-average AP: {ap_macro:.4f}")
    
    return pr_data, img_path, data_path