import jieba
import pandas as pd
import random
import os

CILIN_FILE_PATH = '/mnt/f/google download0/synonyms_dic.txt' 
EMOTION_FILE_PATH = '/mnt/f/google download0/情感词汇本体.xlsx' 

# 全局同义词库变量，键为词语，值为其同义词列表（不包含自己）
GLOBAL_SYNONYMS = {} 

# 全局情感词汇集合，用于随机删除时保护重要词汇
EMOTION_WORDS = set() 

def load_cilin_synonyms(file_path):
    global GLOBAL_SYNONYMS
    
    if not os.path.exists(file_path):
        print(f"错误：同义词文件未找到在路径: {file_path}")
        print("请确保文件存在且路径正确，或者修改 CILIN_FILE_PATH 变量。")
        return

    # 临时字典：键为词林编码，值为该编码下所有的词语集合
    code_to_words = {}
    
    # 关键步骤：使用 encoding='gb18030' 打开文件
    try:
        with open(file_path, 'r', encoding='gb18030') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if '=' in line:
                    parts = line.split('=', 1)
                elif '@' in line:
                    parts = line.split('@', 1)
                else:
                    continue 
                
                code = parts[0].strip()
                words = parts[1].strip().split(' ')
                
                if code not in code_to_words:
                    code_to_words[code] = set()
                
                # 过滤掉空字符串，将词语合并到对应编码下的集合
                code_to_words[code].update(w for w in words if w)
    
    except UnicodeDecodeError:
        print("错误：解码失败。请确认文件编码是否确实是 GB18030。")
        return
    except Exception as e:
        print(f"加载文件时发生其他错误: {e}")
        return


    # 第二遍循环：构建最终的 GLOBAL_SYNONYMS 字典
    synonyms_temp = {}
    for code, all_words_set in code_to_words.items():
        all_words_list = list(all_words_set)
        
        for word in all_words_list:
            # 找到 word 所有的同义词（即 all_words_list 中除了 word 自身以外的所有词）
            synonyms = [w for w in all_words_list if w != word]
            
            if synonyms:
                if word not in synonyms_temp:
                    synonyms_temp[word] = set()
                
                # 保证同义词不重复，并合并来自不同编码的同义词
                synonyms_temp[word].update(synonyms)

    # 将集合转换回列表，并赋值给全局变量
    GLOBAL_SYNONYMS = {word: list(syns) for word, syns in synonyms_temp.items()}
    print(f"同义词库加载完成，共包含 {len(GLOBAL_SYNONYMS)} 个词汇。")

def load_emotion_words(file_path):
    """
    加载大连理工情感词汇本体（.xlsx 文件）并构建情感词集合。
    该集合包含所有褒义词、贬义词和强度词。
    """
    global EMOTION_WORDS
    
    if not os.path.exists(file_path):
        print(f"警告：专业情感词典文件未找到在路径: {file_path}")
        print("退回到使用内置的简化情感词汇列表。")
        # 简化列表作为兜底方案
        EMOTION_WORDS.update({'不', '没', '没有', '非常好', '好', '差', '垃圾', '很', '非常', '更', '最'})
        return

    try:
        df = pd.read_excel(file_path, header=None)
        
        # 假设：0列=词语, 5列=情感极性 (1=褒义, 2=贬义, 0=中性), 4列=情感强度
        WORD_COL = 0
        POLARITY_COL = 5 
        INTENSITY_COL = 4 

        # 提取所有非中性的词语 (极性 != 0) AND 强度大于1的词语 (程度副词)
        emotional_words = df[
            (df[POLARITY_COL] != 0) | (df[INTENSITY_COL] > 1) 
        ][WORD_COL].dropna().astype(str).tolist()

        # 确保所有情感词都是唯一的，并更新全局集合
        EMOTION_WORDS.update(w for w in emotional_words if w.strip())
        
        print(f"专业情感词库加载完成，共包含 {len(EMOTION_WORDS)} 个情感相关词汇。")
        
    except Exception as e:
        print(f"加载情感词典时发生错误: {e}")
        print("请检查 Excel 文件格式和路径，并确保已安装 pandas 和 openpyxl。")
        EMOTION_WORDS.update({'不', '没', '非常好', '好', '差', '垃圾', '很', '非常', '更', '最'})

def get_synonyms(word):
    """从全局同义词库中获取同义词列表"""
    # 从全局字典中获取同义词，找不到返回空列表 []
    return GLOBAL_SYNONYMS.get(word, [])

def synonym_replacement(review, n=1):
    """同义词替换增强 (保持不变)"""
    words = jieba.lcut(review)
    new_words = words.copy()
    
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms_word = get_synonyms(random_word)
        if synonyms_word:
            synonym = random.choice(synonyms_word)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n:
            break

    return ''.join(new_words) 

def random_insertion(review, n=1):
    """随机插入同义词增强 """
    words = jieba.lcut(review)
    new_words = words.copy()
    
    for _ in range(n):
        if not new_words: 
            break
        
        reference_word = random.choice(new_words)
        synonyms = get_synonyms(reference_word)
        
        if synonyms:
            insert_word = random.choice(synonyms)
        else:
            insert_word = reference_word
            
        insert_position = random.randint(0, len(new_words))
        new_words.insert(insert_position, insert_word)
        
    return ''.join(new_words)

def random_deletion(review, p=0.1):
    """
    随机删除词语增强。
    p: 每个非情感词语被删除的概率。
    情感词(EMOTION_WORDS 中的词）将被保留，避免删除。
    """
    words = jieba.lcut(review)
    
    if len(words) <= 1:
        return review

    new_words = []
    # 遍历所有词语
    for word in words:
        # 1. 如果该词是情感态度词，则必须保留
        if word in EMOTION_WORDS:
            new_words.append(word)
            continue
            
        # 2. 如果不是情感词，则根据概率 p 决定是否保留
        #     如果随机数大于 p，则保留该词（即删除概率为 p）
        if random.random() > p:
            new_words.append(word)

    # 如果所有词都被删除了，至少保留原句，以防生成空字符串
    if len(new_words) == 0:
        return review 

    return ''.join(new_words)

if __name__ == '__main__':
    
    # 步骤一：加载同义词库
    load_cilin_synonyms(CILIN_FILE_PATH)
    
    # 步骤二：加载情感词典 
    load_emotion_words(EMOTION_FILE_PATH)

    data_file_path = '/mnt/f/google download0/cleaned_reviews.csv'
    
    try:
        data = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"错误：原始数据文件未找到在路径: {data_file_path}")
        exit()

    TEXT_COLUMN = 'review' 
    if TEXT_COLUMN in data.columns:
        print(f"正在处理 '{TEXT_COLUMN}' 列的缺失值和类型...")
        data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna('').astype(str)
        print("处理完成。")
    else:
        print(f"警告：未找到列名 '{TEXT_COLUMN}'。请检查您的 CSV 列名是否正确。")
        
    print(f"原始数据加载完成，共 {len(data)} 条记录。开始进行数据增强...")

    # 应用数据增强函数
    # 1. 同义词替换 (n=2)
    data['augmented_review'] = data[TEXT_COLUMN].apply(lambda x: synonym_replacement(x, n=2))
    
    # 2. 随机插入 (n=1)
    data['augmented_review'] = data['augmented_review'].apply(lambda x: random_insertion(x, n=1))
    
    # 3. 随机删除 (p=0.1) - 新增
    data['augmented_review'] = data['augmented_review'].apply(lambda x: random_deletion(x, p=0.1))

    # 保存增强后的数据
    output_file_path = '/mnt/f/google download0/test0.csv'
    data.to_csv(output_file_path, index=False)
    
    print(f"数据增强完成，结果已保存至: {output_file_path}")