import re
import string
from collections import Counter

# 停用词（HotpotQA 官方清洗规则，过滤无意义词汇）
STOPWORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when',
    'where', 'how', 'why', 'for', 'of', 'in', 'on', 'at', 'to', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
])

def normalize_answer(s):
    """
    HotpotQA 官方答案归一化函数：清洗答案文本，统一格式
    :param s: 原始答案（黄金答案/生成答案）
    :return: 清洗后的词列表
    """
    def remove_punc(text):
        # 移除所有标点符号
        return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    def lower(text):
        # 转小写
        return text.lower()
    
    def remove_stopwords(tokens):
        # 过滤停用词
        return [token for token in tokens if token not in STOPWORDS and token.strip() != '']
    
    # 执行清洗流程：去标点→转小写→分词→去停用词→去空值
    s = remove_punc(lower(s))
    tokens = s.split()
    tokens = remove_stopwords(tokens)
    return tokens

def compute_em(gold_ans, pred_ans):
    """
    计算 Exact Match（精确匹配）分值
    :param gold_ans: 黄金答案（str）
    :param pred_ans: 模型生成答案（str）
    :return: EM 分值（0 或 1）
    """
    gold_tokens = normalize_answer(gold_ans)
    pred_tokens = normalize_answer(pred_ans)
    return 1 if gold_tokens == pred_tokens else 0

def compute_f1(gold_ans, pred_ans):
    """
    计算 F1 Score（词级别相似度）
    :param gold_ans: 黄金答案（str）
    :param pred_ans: 模型生成答案（str）
    :return: F1 分值（0~1）
    """
    gold_tokens = normalize_answer(gold_ans)
    pred_tokens = normalize_answer(pred_ans)
    
    # 若两者均为空，F1 为 1
    if not gold_tokens and not pred_tokens:
        return 1.0
    # 若其一为空，F1 为 0
    if not gold_tokens or not pred_tokens:
        return 0.0
    
    # 计算词频
    gold_cnt = Counter(gold_tokens)
    pred_cnt = Counter(pred_tokens)
    
    # 计算交集（正确匹配的词数）
    common = gold_cnt & pred_cnt
    num_same = sum(common.values())
    
    # 计算精确率（Precision）、召回率（Recall）
    precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0
    
    # 计算 F1（避免除零）
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def hotpot_qa_evaluate(gold_answers, pred_answers):
    """
    批量评估 HotpotQA 样本，输出整体 EM 和 F1 平均分
    :param gold_answers: 黄金答案列表（list[str]）
    :param pred_answers: 模型生成答案列表（list[str]）
    :return: 字典（平均 EM、平均 F1、单条结果）
    """
    assert len(gold_answers) == len(pred_answers), "黄金答案和生成答案数量必须一致"
    
    total_em = 0.0
    total_f1 = 0.0
    single_results = []  # 存储每条样本的 EM/F1 结果
    
    for gold, pred in zip(gold_answers, pred_answers):
        em = compute_em(gold, pred)
        f1 = compute_f1(gold, pred)
        total_em += em
        total_f1 += f1
        single_results.append({
            "gold_answer": gold,
            "pred_answer": pred,
            "em": em,
            "f1": f1
        })
    
    # 计算平均分
    avg_em = total_em / len(gold_answers) if len(gold_answers) > 0 else 0.0
    avg_f1 = total_f1 / len(gold_answers) if len(gold_answers) > 0 else 0.0
    
    return {
        "average_em": round(avg_em, 4),
        "average_f1": round(avg_f1, 4),
        "single_sample_results": single_results
    }

# ------------------- 测试示例（适配你的数据格式）-------------------
if __name__ == "__main__":
    # 模拟你的样本数据（从你的日志中提取的 case）
    test_cases = [
        {"gold": "Chief of Protocol", "pred": "Chief of Protocol"},  # 正确答案
        {"gold": "Animorphs", "pred": "The Time Quintet"},          # 错误答案
        {"gold": "Greenwich Village, New York City", "pred": "New York"},  # 部分正确
        {"gold": "YG Entertainment", "pred": "Choeun Entertainment"}, # 错误
        {"gold": "Eenasul Fateh", "pred": ""}                         # 无答案
    ]
    
    golds = [case["gold"] for case in test_cases]
    preds = [case["pred"] for case in test_cases]
    
    # 执行评估
    eval_result = hotpot_qa_evaluate(golds, preds)
    
    # 打印结果
    print(f"HotpotQA 整体评估结果：")
    print(f"平均精确匹配（EM）：{eval_result['average_em'] * 100:.2f}%")
    print(f"平均F1相似度：{eval_result['average_f1'] * 100:.2f}%")
    print("\n单条样本详情：")
    for idx, res in enumerate(eval_result["single_sample_results"]):
        print(f"样本{idx+1}：EM={res['em']}, F1={res['f1']:.4f}, 黄金答案：{res['gold_answer']}, 生成答案：{res['pred_answer']}")