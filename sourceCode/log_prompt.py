# 在文件开始处添加导入
import os
import json
from datetime import datetime

# 添加日志记录函数
def log_prompt_response(prompt, response, strategy=None, log_file='./logs/funsearch_llm_original/prompt_response_log.jsonl'):
    """
    记录提示词和响应到日志文件
    
    Args:
        prompt: 发送给模型的提示词
        response: 模型返回的响应
        strategy: 使用的优化策略信息
        log_file: 日志文件路径
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建日志条目
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "response": response,
        "strategy": strategy  # 添加策略信息
    }
    
    # 追加写入日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')