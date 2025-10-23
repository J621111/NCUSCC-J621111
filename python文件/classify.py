import pandas as pd
from sklearn.model_selection import train_test_split
import os


# 文件路径
input_file_path = '/mnt/f/google download0/test0.csv'

# 输出文件名的基础部分
output_base_name = 'test0'

# 输出文件的目录
# 使用 os.path.dirname(input_file_path) 来获取目录，确保文件保存在源文件同级目录
output_dir = os.path.dirname(input_file_path)

# --- 分割比例设置 ---
# 训练集（Train）比例：例如 80%
train_size_ratio = 0.8
# 验证集（Validation）和 测试集（Test）的比例：剩余的 20% 平分，即各占 10%
val_test_ratio = 1 - train_size_ratio # 剩余的 0.2
# 最终的分割比例：Train: 0.8, Validation: 0.1, Test: 0.1

# random_state 用于确保每次运行代码时，分割的结果都是一样的，方便复现
random_seed = 42 


try:
    # 1. 读取 CSV 文件
    df = pd.read_csv(input_file_path)
    print(f"成功读取文件: {input_file_path}")
    print(f"总行数: {len(df)}")
    
    # 2. 第一次分割：分离出训练集 (Train) 和剩余部分 (Temp)
    # Temp 的大小为 1 - train_size_ratio (即 20%)
    df_train, df_temp = train_test_split(
        df, 
        test_size=val_test_ratio, # 剩余部分占总体的比例
        train_size=train_size_ratio, # 训练集占总体的比例
        random_state=random_seed,
        shuffle=True 
    )
    
    # 3. 第二次分割：将剩余部分 (Temp) 平分成验证集 (Validation) 和测试集 (Test)
    # 由于 df_temp 已经是总体的 val_test_ratio，这里 test_size=0.5 意味着它将占 df_temp 的一半
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=0.5, # 注意：这是 df_temp 的 50%，即总体的 10%
        random_state=random_seed,
        shuffle=True 
    )
    
    # 4. 构建输出文件路径
    output_path_train = os.path.join(output_dir, f'{output_base_name}_train.csv')
    output_path_val = os.path.join(output_dir, f'{output_base_name}_val.csv')
    output_path_test = os.path.join(output_dir, f'{output_base_name}_test.csv')

    # 5. 保存分割后的文件
    # index=False 意味着不将 DataFrame 的索引写入 CSV 文件
    df_train.to_csv(output_path_train, index=False)
    df_val.to_csv(output_path_val, index=False)
    df_test.to_csv(output_path_test, index=False)

    print("-" * 30)
    print("分割成功！")
    print(f"最终比例 (Train/Validation/Test): {train_size_ratio} / {val_test_ratio / 2} / {val_test_ratio / 2}")
    print(f"训练集 ({len(df_train)} 行) 保存到: {output_path_train}")
    print(f"验证集 ({len(df_val)} 行) 保存到: {output_path_val}")
    print(f"测试集 ({len(df_test)} 行) 保存到: {output_path_test}")

except FileNotFoundError:
    print(f"错误：文件未找到。请检查路径是否正确: {input_file_path}")
except Exception as e:
    print(f"发生错误: {e}")
    print("请确保文件不是空的，并且是有效的 CSV 格式。")