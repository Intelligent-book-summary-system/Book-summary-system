import pickle
import json

# 读取 pickle 文件
with open('./data/Alice/summaries_reference.pkl', 'rb') as f:
    references = pickle.load(f)

# 将内容转换为 JSON 并保存z
with open('./data/Alice/summaries_reference.json', 'w', encoding='utf-8') as f:
    json.dump(references, f, ensure_ascii=False, indent=2)
