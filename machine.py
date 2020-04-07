import numpy as np
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as hcluster
"""
Первые три пункта - в конце
"""
def word_extractor(txt_name):
    basewords = set()
    extra = set()
    with open (txt_name, 'r', encoding = 'utf-8') as k:
        k = k.readlines()
        for line in k:
            line = line.split('\t')[0].split('_')
            basewords.add(line[0])
            extra.add(line[1])
    return(basewords, extra)
array = []
vector_names=[]
verb_vector=[]
with open ('model.txt', 'r',encoding='utf-8') as k:
    baseword, extra = word_extractor('words.txt')
    for line in k:
        line = line.strip().split(' ')
        vector = []
        a = line[0].split('_')[0]
        if (a in extra) and line[0].split('_')[1] == 'NOUN':
            for item in line:
                if item == line[0]:
                    vector_names.append(item)
                else:
                    vector.append(float(item))
            array.append(vector)
        if a in baseword:
            for item in line:
                if item == line[0]:
                    continue
                else:
                    verb_vector.append(float(item))
            baseword = ''


words = np.array(array)  # общая матрица

verb_proper_vector = np.array(verb_vector)  # вектор глагола
words = words + verb_proper_vector

Z = hcluster.linkage(words)
dn = hcluster.dendrogram(Z,labels=np.array(vector_names))
clusters = hcluster.fcluster(Z,1)

print(clusters)  # кластеры методом иерархической кластеризации
kmeans = KMeans(n_clusters=6).fit(words)
print(kmeans.labels_)  # кластеры методом К-средних
np.savetxt('full_matrix.txt',words)
np.savetxt('verb_vector.txt',verb_proper_vector)
np.savetxt('Dclusters.txt',clusters)
np.savetxt('Kclusters.txt',kmeans.labels_)