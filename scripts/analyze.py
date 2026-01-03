import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Config
# =====================
OUTPUT_DIR = "figures"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.style.use("default")

# =====================
# Load data
# =====================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# =====================
# Length computation
# =====================
for df in [train_df, test_df]:
    df["title_len"] = df["title"].astype(str).apply(len)
    df["desc_len"] = df["description"].astype(str).apply(len)

# =========================================================
# TRAIN SET ANALYSIS
# =========================================================

# ---- Overall title length
plt.figure(figsize=(6, 4))
sns.histplot(train_df["title_len"], bins=50, kde=True)
plt.title("Train Set Title Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "title_length_distribution.png"), dpi=300)
plt.close()

# ---- Overall description length
plt.figure(figsize=(6, 4))
sns.histplot(train_df["desc_len"], bins=100, kde=True)
plt.title("Train Set Description Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "description_length_distribution.png"), dpi=300)
plt.close()

# ---- Title length by category
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=train_df,
    x="categories",
    y="title_len"
)
plt.title("Train Set Title Length by Category")
plt.xlabel("Category")
plt.ylabel("Number of Characters")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "title_length_by_category.png"), dpi=300)
plt.close()

# ---- Description length by category
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=train_df,
    x="categories",
    y="desc_len"
)
plt.title("Train Set Description Length by Category")
plt.xlabel("Category")
plt.ylabel("Number of Characters")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "description_length_by_category.png"), dpi=300)
plt.close()

# =========================================================
# TEST SET ANALYSIS (NO CATEGORY)
# =========================================================

# ---- Test title length
plt.figure(figsize=(6, 4))
sns.histplot(test_df["title_len"], bins=50, kde=True)
plt.title("Test Set Title Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(TEST_DIR, "title_length_distribution.png"), dpi=300)
plt.close()

# ---- Test description length
plt.figure(figsize=(6, 4))
sns.histplot(test_df["desc_len"], bins=100, kde=True)
plt.title("Test Set Description Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(TEST_DIR, "description_length_distribution.png"), dpi=300)
plt.close()

print("EDA finished. Figures saved to ./figures/")
