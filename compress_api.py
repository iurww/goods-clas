import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def build_prompt(title, description, max_output_length=80):
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
1. Extract key information, first convert the title to a concise item name
2. Remove marketing fluff and promotional language and specific specification parameters
3. Maximum {max_output_length} words
4. Output must be concise and informative

Compressed Text:"""
        
    return prompt

def call_llm_api(client, prompt, model="qwen-plus", max_retries=3):
    """调用大模型API (使用OpenAI SDK)"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            # print(result)

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(1)  # 等待1秒后重试
            else:
                print(f"API调用最终失败: {str(e)}")
                return None

    return None

def evaluate_single_row(client, row, model):
    """评估单行数据"""
    try:
        eval_prompt = build_prompt(
            row['title'],
            row['description'],
            max_output_length=80
        )

        compress = call_llm_api(client, eval_prompt, model)

        if compress is None:
            print(f"压缩第 {row['id']} 空")
            return {
                'id': row['id'],
                'description': row['title'] + row['description'],
                'label': row['categories'],
                'index': row.name
            }

        return {
            'id': row['id'],
            'description': compress,
            'label': row['categories'],
            'index': row.name
        }
    except Exception as e:
        print(f"压缩第 {row['id']} 行时出错 in single: {str(e)}")
        return {
            'id': row['id'],
            'description': -1,
            'label': row['categories'],
            'index': row.name
        }

def evaluate_responses(input_csv, output_csv, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model="qwen-plus", sample_size=None, max_workers=10):
    """
    并发评估回复质量

    参数：
    - input_csv: 输入CSV文件路径
    - output_csv: 输出CSV文件路径
    - api_key: API密钥
    - base_url: API基础URL (默认为阿里云通义千问)
    - model: 模型名称 (默认为qwen-plus)
    - sample_size: 可选，只处理前N条数据（用于测试）
    - max_workers: 并发线程数 (默认10)
    """

    print("正在读取数据...")
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False, quotechar='"', engine="python")

    # 数据预处理
    if sample_size:
        df = df.head(sample_size)

    print(f"共需评估 {len(df)} 条数据")
    print(f"使用模型: {model}")
    print(f"并发线程数: {max_workers}")

    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    results = []

    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(evaluate_single_row, client, row, model): row
            for _, row in df.iterrows()
        }

        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                row = futures[future]
                print(f"处理第 {row['id']} 行时发生异常: {str(e)}")
                results.append({
                    'id': row['id'],
                    'description': -1,
                    'label': row['categories'],
                    'index': row.name
                })

    # 按原始顺序排序
    results.sort(key=lambda x: x['index'])

    # 创建结果DataFrame (移除index列)
    result_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'index'} for r in results])

    # 保存结果
    result_df.to_csv(output_csv, index=False)

    print(f"\n评估完成！结果已保存至: {output_csv}")

    return result_df

if __name__ == "__main__":

    INPUT_FILE = "./data/train.csv"
    OUTPUT_FILE = "./data/compress_train.csv"

    API_KEY = "sk-4f4499ad108a440aafa352e6b25b64a6"

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL = "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max

    print("开始小样本测试...")
    # result_df = evaluate_responses(
    #     INPUT_FILE,
    #     OUTPUT_FILE,
    #     api_key=API_KEY,
    #     base_url=BASE_URL,
    #     model=MODEL,
    #     sample_size=50,
    #     max_workers=5  # 测试时使用较少的并发数
    # )

    # 如果测试通过，取消注释进行完整评估
    print("\n开始完整评估...")
    result_df = evaluate_responses(
        INPUT_FILE,
        OUTPUT_FILE,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        max_workers=20  # 可以根据API限流情况调整，建议10-20
    )

    print("\n前10条结果:")
    print(result_df.head(10))