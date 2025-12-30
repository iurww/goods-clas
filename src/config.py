import torch

class Config:
    # 路径配置
    train_csv = 'data/compress_train.csv'
    test_csv = 'data/compress_test.csv'
    test_probs = 'data/submission_probs.csv'
    train_images_dir = 'data/train_images/train_images'
    test_images_dir = 'data/test_images/test_images'
    submission_file = 'data/submission.csv'
    
    # 模型配置 - 使用SigLIP模型
    model_name = './models/siglip-so400m-patch14-384'
    
    num_classes = 21
    hidden_dim = 1536  # 分类头隐藏层维度
    dropout = 0.2
    
    # 训练配置
    batch_size = 8
    eval_batch_size = 16
    num_epochs = 5
    learning_rate = 2e-5
    weight_decay = 0.05
    warmup_ratio = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    n_folds = 5
    
    # SigLIP文本最大长度上限
    max_text_length = 64
    
    # 图像配置
    image_size = 384
    
    # 是否使用类别权重
    use_class_weight = False
    
    # 混合精度训练
    use_fp16 = True
    
    skip_folds = 0