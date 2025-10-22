import pandas as pd 
import re
import warnings

# 忽略可能出现的警告
warnings.filterwarnings("ignore", category=FutureWarning)

file_path_base = '/mnt/f/google download0/'
file_path = file_path_base + '训练集.csv'
stopwords_file_path = file_path_base + 'cn_stopwords.txt'

# 尝试从外部文件加载中文停用词
try:
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        # 读取所有行，去除两端空白，并使用 set 去重
        chinese_stop_words = set(line.strip() for line in f if line.strip())
    print(f"成功从文件 '{stopwords_file_path}' 加载 {len(chinese_stop_words)} 个停用词。")
except FileNotFoundError:
    print(f"警告: 未找到停用词文件 '{stopwords_file_path}'。将使用一个精简的内置列表代替。")
    # 如果找不到文件，则回退到一个内置的精简列表
    chinese_stop_words = set([
        '的', '地', '得', '是', '有', '一', '在', '个', '这', '那', '之', '了', 
        '和', '也', '而', '给', '被', '对', '于', '与', '或', '但', '还', '都', 
        '又', '很', '更', '最', '就', '来', '去', '说', '做', '看', '让', '把', 
        '我', '你', '他', '她', '它', '们', '呢', '嘛', '吧', '啊', '哦', '呀', 
        '啦', '哇', '哈', '嘿', '才', '将', '向', '比', '如', '而且', '但是', '所以', 
    ])

# 读取数据
data = pd.read_csv(file_path)

# 将包含评论内容的列 '评论内容' 重命名为 'review'
data.rename(columns={'评论内容': 'review'}, inplace=True)
print("列名 '评论内容' 已成功更名为 'review'。")


# 处理数据集使其变成三分类
def map_to_three_categories(rating):
    """根据评分将评价映射到三分类标签。"""
    if rating < 3:
        return 0  # 负面 
    elif rating == 3:
        return 1  # 中性 
    elif rating > 3:
        return 2  # 正面 
    else:
        # 处理可能的缺失值或非预期的评分
        return -1

# 创建新的三分类标签列
data['label'] = data['评分'].apply(map_to_three_categories)

# 检查新的标签分布
print("\n三分类标签的分布:")
print(data['label'].value_counts())

# 显示处理后的数据框的前几行
print("\n带有三分类标签的数据框前5行:")
print(data[['评分', 'label', '评论标题', 'review']].head())

# 删除原始的 '评分' 列（如果只需要新标签）
data = data.drop('评分', axis=1)

# 处理缺失值
data.dropna(subset=['review'], inplace=True)
data['review'] = data['review'].astype(str)


# 去除特殊字符
def remove_special_characters(text):
    # 保留字母、数字和中文字符
    return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', text)

data['review'] = data['review'].apply(remove_special_characters)


# 去除停用词
def remove_stopwords(text, stop_words=chinese_stop_words):
    # 构建停用词匹配模式 (使用 '|' 连接，实现“或”操作)
    pattern = '|'.join(re.escape(word) for word in stop_words if word.strip())
    
    if pattern:
        # 移除停用词
        cleaned_text = re.sub(pattern, '', text)
    else:
        cleaned_text = text
        
    # 移除多余的空格
    return re.sub(r'\s+', '', cleaned_text).strip()

data['review'] = data['review'].apply(remove_stopwords)


# 去除重复数据
data.drop_duplicates(subset=['review'], inplace=True)

# 删除不必要的字符
def remove_unnecessary_chars(review):
    # 删除特定中文词汇 (保留您的原始逻辑)
    pattern = re.compile(r'这本书|我|的|这|们|一个|一本', re.IGNORECASE)
    cleaned_text = pattern.sub('', review)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

data['review'] = data['review'].apply(remove_unnecessary_chars)

# 删除过长评论
max_length = 50
print(f"\n原始数据量: {len(data)}")
data = data[data['review'].str.len() <= max_length]
print(f"删除过长评论后数据量 (<= {max_length}): {len(data)}")

# 统一编码
def unify_encoding(text):
    return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

data['review'] = data['review'].apply(unify_encoding)

# 保存清洗后的数据
cleaned_file_path = file_path_base + 'cleaned_reviews.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\n清洗后的数据已保存到 '{cleaned_file_path}'。")

# 输出清洗后的数据预览
print("\n--- 清洗后的数据预览(前10行)---")
print(data.head(10).to_string())
