import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os

class QwenTextCompressor:
    def __init__(self, model_path, device='cuda', batch_size=8):
        """
        初始化Qwen模型压缩器
        
        参数:
            model_path: Qwen模型本地路径
            device: 运行设备 (cuda/cpu)
            batch_size: 批处理大小
        """
        print(f"Loading Qwen model from {model_path}...")
        self.device = device
        self.batch_size = batch_size
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        print(f"Model loaded on {device}")
        
        # 类别描述（英文）
        self.category_descriptions = {
            0: "Traditional consumer electronics accessories and photography storage accessories",
            1: "Outdoor sports and leisure products: hunting, fishing, camping, water sports, ball games, fitness, cycling, martial arts, shooting range",
            2: "Mobile phones and related accessories",
            3: "Automotive and auto parts",
            4: "Board games and educational toys",
            5: "Hardware, building materials and tools",
            6: "Health and personal care products",
            7: "Daily chemical, wash and beauty products",
            8: "Food and beverage products",
            9: "Office and school supplies",
            10: "Handicraft/art creation materials and tools",
            11: "Pet supplies",
            12: "Outdoor/patio and gardening products",
            13: "Clothing and accessories",
            14: "Baby durable goods/hardware category",
            15: "Musical instruments and audio equipment",
            16: "Daily necessities/consumables/small commodities",
            17: "Baby consumables/care products + children's daily care",
            18: "Home appliance accessories and consumables",
            19: "High-end skincare + professional makeup + salon/medical aesthetics + niche fragrance + adult products",
            20: "Electronic product accessories + smart peripherals + digital home"
        }
    
    def build_prompt(self, title, description, category_id, max_output_length=80):
        """
        构建压缩提示词（英文）
        
        设计原则:
        1. 明确任务目标
        2. 提供类别上下文
        3. 限制输出长度
        4. 要求保留关键信息
        """
        prompt = f"""Task: Extract the most relevant information from product text for classification.

Product Title: {title}

Product Description: {description}

Instructions:
1. Extract key information most relevant to the category
2. Remove marketing fluff and promotional language and specific specification parameters
3. Maximum {max_output_length} words
4. Output must be concise and informative

Compressed Text:"""
        
        return prompt
    
    def compress_single(self, title, description, category_id, max_output_length=80):
        """
        压缩单条文本
        """
        prompt = self.build_prompt(title, description, category_id, max_output_length)
        
        # 使用chat格式
        messages = [
            {"role": "system", "content": "You are a text summarization assistant. Extract key product information concisely."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_output_length * 2,  # 字符数约为单词数的1.5-2倍
                temperature=0.3,  # 低温度保证稳定性
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 清理输出
        response = response.strip()
        # 移除可能的重复前缀
        if response.startswith("Compressed Text:"):
            response = response.replace("Compressed Text:", "").strip()
        
        return response
    
    def compress_batch(self, titles, descriptions, categories, max_output_length=80):
        """
        批量压缩（简化版，单条处理避免复杂的batch对齐）
        """
        results = []
        for title, desc, cat in zip(titles, descriptions, categories):
            try:
                compressed = self.compress_single(title, desc, cat, max_output_length)
                results.append(compressed)
            except Exception as e:
                print(f"Error processing: {e}")
                # 失败时返回截断的原文
                results.append(f"{title} {desc}"[:max_output_length * 6])
        
        return results
    
    def compress_dataset(self, input_csv, output_csv, max_output_length=80, save_every=500):
        """
        压缩整个数据集
        
        参数:
            input_csv: 输入CSV路径
            output_csv: 输出CSV路径
            max_output_length: 压缩后的最大单词数
            save_every: 每处理多少条保存一次（防止中断丢失）
        """
        df = pd.read_csv(input_csv)
        
        # 填充缺失值
        df['title'] = df['title'].fillna('')
        df['description'] = df['description'].fillna('')
        
        print(f"Starting compression for {len(df)} samples...")
        print(f"Batch size: {self.batch_size}")
        
        compressed_texts = []
        
        # 批量处理
        for i in tqdm(range(0, len(df), self.batch_size), desc="Compressing"):
            batch_df = df.iloc[i:i+self.batch_size]
            
            batch_results = self.compress_batch(
                batch_df['title'].tolist(),
                batch_df['description'].tolist(),
                batch_df['categories'].tolist(),
                max_output_length
            )
            
            for r in batch_results:
                print(f"Compressed{len(r)}: {r}...")
            
            compressed_texts.extend(batch_results)
            
            # 定期保存
            if (i + self.batch_size) % save_every == 0:
                temp_df = df.iloc[:len(compressed_texts)].copy()
                temp_df['compressed_text'] = compressed_texts
                temp_df.to_csv(output_csv + '.tmp', index=False)
                print(f"\nCheckpoint saved: {len(compressed_texts)} samples")
        
        # 最终保存
        df['compressed_text'] = compressed_texts
        df['original_length'] = df['title'].str.len() + df['description'].str.len()
        df['compressed_length'] = df['compressed_text'].str.len()
        
        df.to_csv(output_csv, index=False)
        
        # 删除临时文件
        if os.path.exists(output_csv + '.tmp'):
            os.remove(output_csv + '.tmp')
        
        # 统计
        avg_original = df['original_length'].mean()
        avg_compressed = df['compressed_length'].mean()
        compression_rate = (1 - avg_compressed / avg_original) * 100
        
        print(f"\n{'='*50}")
        print(f"Compression completed!")
        print(f"Original avg length: {avg_original:.1f} characters")
        print(f"Compressed avg length: {avg_compressed:.1f} characters")
        print(f"Compression rate: {compression_rate:.1f}%")
        print(f"Saved to: {output_csv}")
        
        return df
    
    def compare_samples(self, df, n_samples=5):
        """
        对比展示压缩效果
        """
        print(f"\n{'='*80}")
        print("Sample Compression Results:")
        print('='*80)
        
        sample_df = df.sample(n=min(n_samples, len(df)))
        
        for idx, row in sample_df.iterrows():
            print(f"\nCategory: {row['categories']}")
            print(f"Original ({row['original_length']} chars):")
            print(f"  Title: {row['title'][:100]}")
            print(f"  Desc: {row['description'][:200]}...")
            print(f"\nCompressed ({row['compressed_length']} chars):")
            print(f"  {row['compressed_text']}")
            print("-" * 80)


# ============ 使用脚本 ============
if __name__ == "__main__":
    # 配置
    MODEL_PATH = "../human-preference/models/qwen3-4b"  # 修改为你的模型路径
    INPUT_CSV = "data/train.csv"
    OUTPUT_CSV = "data/train_compressed_qwen.csv"
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 初始化压缩器
    compressor = QwenTextCompressor(
        model_path=MODEL_PATH,
        device=device,
        batch_size=8 if device == 'cuda' else 2  # GPU用8，CPU用2
    )
    
    # 批量压缩
    result_df = compressor.compress_dataset(
        INPUT_CSV,
        OUTPUT_CSV,
        max_output_length=80,  # 控制输出在80个单词左右
        save_every=500
    )
    
    # 展示样例
    compressor.compare_samples(result_df, n_samples=5)
    
    print("\n✅ All done! You can now use the compressed text for training.")