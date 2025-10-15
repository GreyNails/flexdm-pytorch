import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import pandas as pd

# 读取 tfevents 文件
log_file = '/home/dell/Project-HCL/BaseLine/flexdm_pt/scripts/logs/retrain/events.out.tfevents.1760457882.29541d69cb02.2298748.0'

# 解析事件数据
data = {'step': [], 'tag': [], 'value': []}

for event in summary_iterator(log_file):
    for value in event.summary.value:
        data['step'].append(event.step)
        data['tag'].append(value.tag)
        data['value'].append(value.simple_value)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 可视化不同的指标
tags = df['tag'].unique()

fig, axes = plt.subplots(len(tags), 1, figsize=(12, 4*len(tags)))
if len(tags) == 1:
    axes = [axes]

for idx, tag in enumerate(tags):
    tag_data = df[df['tag'] == tag]
    axes[idx].plot(tag_data['step'], tag_data['value'])
    axes[idx].set_title(tag)
    axes[idx].set_xlabel('Step')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("训练指标统计：")
print(df.groupby('tag')['value'].describe())