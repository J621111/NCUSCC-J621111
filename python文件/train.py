import os
import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    pipeline
)

# 0. 配置和设置
# 模型名称：使用中文BERT基础模型
MODEL_NAME = 'bert-base-chinese' 
# 分类标签数量（0: 负面, 1: 中立 2：正面）
NUM_LABELS = 3
# 最大序列长度
MAX_LENGTH = 128 
# 训练超参数
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
# 输出目录，用于保存模型检查点和最终模型。
OUTPUT_DIR = "/mnt/f/google download0/bert_sentiment_results"

TEXT_COLUMN_NAME = 'augmented_review' 

# 模型训练所需列必须包含分词器输出的 input_ids, attention_mask, token_type_ids
MANDATORY_TRAINER_COLUMNS = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
COLUMNS_TO_KEEP = ['数据ID', TEXT_COLUMN_NAME]

# 最终数据集将包含 BERT 模型的输入列、标签列，辅助列
MODEL_INPUT_COLUMNS = MANDATORY_TRAINER_COLUMNS + COLUMNS_TO_KEEP

# 辅助列，用于在 set_format 前检查或移除 
AUX_COLUMNS = ['数据ID', TEXT_COLUMN_NAME]


# 1. 实际数据加载和准备 (使用 CSV 文件)
# 期望加载的 CSV 文件路径
DATA_FILES = {
    # 训练集
    "train": '/mnt/f/google download0/test0_train.csv',
    # 验证集
    "validation": '/mnt/f/google download0/test0_val.csv',
    # 测试集
    "test": '/mnt/f/google download0/test0_test.csv'
    }

# 临时变量，用于处理数据加载失败时的情况
current_text_column_name = TEXT_COLUMN_NAME 

try:
    # 从 CSV 文件加载数据
    raw_datasets = load_dataset("csv", data_files=DATA_FILES)
    
    print("--- 实际数据集加载成功 ---")
    print(raw_datasets)
    print(f"训练集数据量: {len(raw_datasets['train'])}")
    
except Exception as e:
    print(f"警告：数据加载失败，请检查 CSV 文件路径和格式：{e}")
    print(f"预期文件路径: {', '.join(DATA_FILES.values())} 缺失或格式错误。")
    
    # 如果加载失败，使用小型模拟数据进行演示
    print("--- 警告：使用模拟数据进行演示，请勿用于正式训练！ ---")
    data = {
        # 模拟数据使用 'augmented_review' 作为文本列
        'augmented_review': [
            "这个手机太棒了，完全超出预期！", 
            "收到货，颜色不对，有点失望。", 
            "物流速度很快，但商品本身中规中矩。",
            "服务态度很好，下次还会再来。",
            "这个小吃味道一般，不值得推荐。",
            "性价比很高，非常满意的一次购物。"
        ] * 10, 
        'label': [2, 0, 1, 2, 0, 2] * 10, 
        # 模拟添加一个ID列
        '数据ID': [f'ID_{i}' for i in range(60)]
    }
    df_all = pd.DataFrame(data)
    
    # 划分数据集
    df_train, df_temp = train_test_split(df_all, test_size=0.4, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(df_train, preserve_index=False),
        "validation": Dataset.from_pandas(df_val, preserve_index=False),
        "test": Dataset.from_pandas(df_test, preserve_index=False)
    })
    
    # 如果是模拟数据，临时修改当前使用的文本列名以匹配模拟数据
    current_text_column_name = 'augmented_review' 
    
    print(f"已创建模拟训练集大小: {len(raw_datasets['train'])}")
    print(f"当前使用的文本列名: {current_text_column_name}")


# 2. 数据预处理（Tokenization）

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    """
    对文本进行分词、填充和截断。
    """
    # 使用当前生效的文本列名访问文本列
    # 确保所有输入都被转换为字符串，避免潜在的 NaN/None 错误
    review = [str(x) if x is not None else "" for x in examples[current_text_column_name]] 
    
    return tokenizer(review, 
                     padding='max_length', 
                     truncation=True, 
                     max_length=MAX_LENGTH)

# 应用预处理
tokenized_datasets = {}
for split, dataset in raw_datasets.items():
    
    # 过滤掉标签为 None/NaN 的样本 
    print(f"--- {split}数据集原始大小: {len(dataset)}")
    
    # 动态确定标签列名
    label_column_name = 'label' if 'label' in dataset.column_names else (
        'labels' if 'labels' in dataset.column_names else None
    )
    
    if label_column_name:
        dataset = dataset.filter(
            lambda example: example.get(label_column_name) is not None and not (isinstance(example.get(label_column_name), float) and np.isnan(example.get(label_column_name))),
            load_from_cache_file=False
        )
        print(f"--- {split}数据集移除缺失标签后大小: {len(dataset)}")
    
    # 1. 应用分词函数，生成 'input_ids', 'attention_mask', 'token_type_ids'
    tokenized_datasets[split] = dataset.map(
        preprocess_function, 
        batched=True, 
        load_from_cache_file=False 
    )
    
    # 2. 将标签列重命名为模型所需的 'labels' 
    # 确保标签列名统一为 'labels'
    if 'label' in tokenized_datasets[split].column_names:
        tokenized_datasets[split] = tokenized_datasets[split].rename_column("label", "labels")
    
    # 3. 确保 labels 必须是 torch.long 
    # 因为 Trainer 要求 'labels' 必须是一个张量或可转换为张量的类型，且通常是 long/int
    tokenized_datasets[split] = tokenized_datasets[split].map(
        lambda example: {'labels': torch.tensor(example['labels']).long()},
        batched=True, 
        load_from_cache_file=False
    )
    
    # 获取当前数据集中的所有列名
    all_columns = tokenized_datasets[split].column_names
    
    # 保留 MODEL_INPUT_COLUMNS 中包含的列
    columns_to_remove = [col for col in all_columns if col not in MODEL_INPUT_COLUMNS]

    print(f"将移除的列: {columns_to_remove}")
    
    # 4. 移除多余的列
    # 这一步是为了精简数据集，只保留模型输入和标签
    tokenized_datasets[split] = tokenized_datasets[split].remove_columns(columns_to_remove)
    
    # 5. 设置数据格式为 PyTorch tensors
    tokenized_datasets[split].set_format(
        "torch", 
        columns=MODEL_INPUT_COLUMNS, # 现在包含了所有必需的输入张量和用户想保留的辅助列
        output_all_columns=False
    )

    print(f"{split} 数据集预处理完成，包含列: {tokenized_datasets[split].column_names}")

# 3. 定义评估指标
# 加载评估指标工具，使用 F1-score 作为主要指标
metric_f1 = evaluate.load("f1") 
metric_acc = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """计算分类指标 (F1-macro 和 Accuracy)"""
    logits, labels = eval_pred
    # 使用 numpy.argmax 从 logits 中获取预测类别
    predictions = np.argmax(logits, axis=-1)
    
    # 计算 F1 (macro average)，适用于多分类任务
    f1_score = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    
    # 额外计算 Accuracy
    accuracy = metric_acc.compute(predictions=predictions, references=labels)

    # 返回字典格式的结果
    return {**f1_score, **accuracy}

# 4. 模型加载和微调

print(f"\n--- 正在加载模型: {MODEL_NAME} (标签数: {NUM_LABELS}) ---")

# 加载预训练模型，并添加分类头
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS, 
    # 映射标签ID到人类可读的名称 
    id2label={0: "负面", 1:"中立", 2: "正面"},
    label2id={"负面": 0, "中立": 1, "正面": 2}
)

# 定义最终模型保存路径，位于 OUTPUT_DIR 之下
final_model_path = os.path.join(OUTPUT_DIR, "final_sentiment_model")

# 设置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, # 输出目录（检查点也保存在此）
    num_train_epochs=NUM_EPOCHS, # 训练轮次

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE, # 学习率
    weight_decay=0.01,
    eval_strategy="epoch", # 每个 epoch 结束时评估
    save_strategy="epoch", # 每个 epoch 结束时保存
    load_best_model_at_end=True, # 训练结束时加载验证集上表现最好的模型
    metric_for_best_model="eval_f1", # 根据 F1-score 确定最佳模型
    greater_is_better=True,
    logging_dir='./logs',
    report_to="none", # 在非Jupyter环境中禁用报告到wandb等
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练
print("\n--- 开始模型微调 ---")
try:
    trainer.train()
    print("--- 模型微调完成 ---")
except Exception as e:
    print(f"训练过程中发生错误: {e}")

# 5. 最终评估与保存

# 最终评估测试集
print("\n--- 测试集最终评估 ---")
eval_results = trainer.evaluate(tokenized_datasets["test"])
print(eval_results)

# 保存最终模型和分词器
# 确保目录存在
os.makedirs(final_model_path, exist_ok=True)
# Trainer 会将最佳模型和分词器保存到指定路径
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\n--- 最佳模型已保存到: {final_model_path} ---")


# 6. 模型推理演示
print("\n--- 加载最终模型进行推理演示 ---")

# 创建预测 Pipeline，指定使用保存的路径
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=final_model_path,
    tokenizer=final_model_path
)

# 示例预测
test_comments = [
    "虽然价格贵了点，但是质量确实没得说，好评！",
    "差评，用了两天就坏了，质量堪忧。",
    "东西和描述一致，没有惊喜也没有失望。",
    "客服态度差到极点，再也不会来了。",
    "整体体验超出我的预期，非常惊喜。"
]

print("\n--- 预测结果示例 ---")
for comment in test_comments:
    # 进行预测
    result = sentiment_pipeline(comment)[0]
    
    # 获取模型输出的标签 (如: '负面', '中立', '正面')
    predicted_label = result['label']
    
    print(f"评价: {comment}")
    print(f" 预测情感: {predicted_label} (分数: {result['score']:.4f})")
print("\n--- 推理演示完成 ---")
