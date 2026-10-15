import json
import csv

# 读取JSON文件
with open('./results/qiprune_sweep.json', 'r') as f:
    data = json.load(f)

# 获取前32条结果
results = data['results'][:32]

# 如果结果不足32条，给出提示
if len(results) < 32:
    print(f"注意：JSON文件中只有{len(results)}条结果，将全部转换")

# 提取所有可能的字段（确保所有键都被包含）
fields = set()
for result in results:
    fields.update(result.keys())
fields = sorted(fields)

# 写入CSV文件
with open('./results/qiprune_sweep_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print("转换完成，已生成qiprune_sweep_results.csv文件")