import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adata_set import dataset_process
from scDCC import scDCC
from utils import cluster_acc
from sklearn import metrics

adata = dataset_process('data/CITE_CBMC_counts_top2000.h5').adata

model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=15, encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=2.5, gamma=1).cuda()
checkpoint = torch.load('results/scDCC_p0_1/FTcheckpoint_2.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
labeled = np.loadtxt('label_selected_cells_1.txt', dtype=int)
x = adata.X[labeled,:]
y = adata.obs['Group'][labeled]
x = torch.from_numpy(x)
q, _, _, _, _ = model.forward(x.to('cuda'))
q = q.detach().cpu().numpy()
y_pred = np.argmax(q, axis=1)
acc = np.round(cluster_acc(y, y_pred), 5)
nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
"""
方法1：PCA线性降维
"""
# 创建PCA对象，并指定目标维度为2
pca = PCA(n_components=2)
# 执行PCA降维
reduced_data = pca.fit_transform(q)

"""
方法2：TSNE非线性降维
"""
# 创建TSNE对象，并指定目标维度为2
tsne = TSNE(n_components=2)
# 执行TSNE降维
reduced_data2 = tsne.fit_transform(q)
# 获取不同类别的唯一值
unique_labels = np.unique(y).astype(int)
length = len(unique_labels)
# 定义颜色和标记列表
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'mediumaquamarine', 'lightcoral',
          'khaki', 'mediumpurple', 'darksalmon']
markers = ['o', 's', '^', 'v', 'D', 'P', '*', '+', 'x', '.', '>', '<', 'h', '8', 'd']

# 针对每个类别，使用不同的颜色绘制数据点
for i, label in enumerate(unique_labels):
    class_data = reduced_data2[y == label]
    plt.scatter(class_data[:,0],class_data[:,1], c=colors[i], marker=markers[i], label=f'Class {i}')
plt.legend(prop={'size':10}, loc='upper right', bbox_to_anchor=(1.3, 1))
plt.subplots_adjust(right=0.8)

plt.text(0, 65, 'ACC:%.3f, NMI:%.3f, ARI:%.3f'%(acc, nmi, ari), ha='center', va='bottom')
plt.show()