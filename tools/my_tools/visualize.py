import torch
import matplotlib.pyplot as plt

# 加载场景图
result_dict = torch.load("C:\\0\\vg\\2023-03-18_10-59-(top1)VG-sgtr-dec_layer6\\result_dict.pytorch")


# 从result_dict中获取关系预测的精度和召回率
rel_precision = result_dict['sgdet_accuracy_hit']
rel_recall = result_dict['sgdet_recall']

# 绘制精度-召回率曲线
fig, ax = plt.subplots()
ax.plot(rel_recall, rel_precision)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True)
plt.title('Precision-Recall Curve for Relationship Detection')
plt.show()





