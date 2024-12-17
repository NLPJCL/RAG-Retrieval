import json

# 将 [query, content, score] 转换为 {"query": str(required), "pos": List[str], "pos_scores": List[float]} 格式

data_list = []
with open('t2rank_100.distill.jsonl') as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)

grouped_data = {}
for data in data_list:
    if data['query'] not in grouped_data:
        grouped_data[data['query']] = []
    grouped_data[data['query']].append(data)

with open("t2rank_100.distill.standard.jsonl", "w") as f:
    for query in grouped_data:
        adict = dict()
        adict['query'] = query
        adict['pos'] = []
        # adict['neg'] = []
        adict["pos_scores"] = []
        # adict["neg_scores"] = []
        for doc in grouped_data[query]:
            adict['pos'].append(doc['content'])
            adict['pos_scores'].append(doc['score'])
        f.write(json.dumps(adict, ensure_ascii=False) + "\n")
        