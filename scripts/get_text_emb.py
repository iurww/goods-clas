import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================= 配置 =================
MODEL_DIR = "./models/qwen3-4b"   # 你的 Qwen3-4B 本地目录
CSV_PATH = "data/test.csv"
OUTPUT_PATH = "data/qwen3_test_text_embs.npy"

BATCH_SIZE = 1          # 4B 模型，16 已经比较安全
MAX_LEN = 256            # 商品描述一般不需要太长
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16    # 节省显存

# ================= 加载模型 =================
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_DIR,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ================= 读取数据 =================
df = pd.read_csv(CSV_PATH)
texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()

# ================= mean pooling =================
def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

# ================= 编码 =================
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )

        # 取最后一层 hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, T, D]

        emb = mean_pool(last_hidden, inputs.attention_mask)  # [B, D]

        # 可选：L2 normalize（推荐）
        emb = torch.nn.functional.normalize(emb, dim=1)

        all_embeddings.append(emb.cpu().numpy())

# ================= 保存 =================
all_embeddings = np.concatenate(all_embeddings, axis=0)
np.save(OUTPUT_PATH, all_embeddings)

print(f"Saved embeddings: {all_embeddings.shape} → {OUTPUT_PATH}")
