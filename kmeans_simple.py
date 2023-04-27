import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터프레임(x, y)
df = pd.DataFrame(columns=['x', 'y'])

# 데이터 입력
df.loc[0] = [3,1]
df.loc[1] = [4,1]
df.loc[2] = [3,2]
df.loc[3] = [4,2]
df.loc[4] = [10,5]
df.loc[5] = [10,6]
df.loc[6] = [11,5]
df.loc[7] = [11,6]
df.loc[8] = [15,1]
df.loc[9] = [15,2]
df.loc[10] = [16,1]
df.loc[11] = [16,2]

# x축, y축 설정, 출력할 데이터 설정, 선형회귀선, 점크기
sns.lmplot(x = "x", y = "y", data=df, fit_reg=False, scatter_kws={"s": 200})
plt.title('kmeans plot')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

# df는 데이터프레임(엑셀과 같은 행렬)
# data_points는 배열형태
# K-means(가장 가까운) 클러스터 개수, 첫 centroid, 데이터입력
data_points = df.values
kmeans = KMeans(n_clusters=3, n_init='auto').fit(data_points)

# labels_로 클러스터 표시
# cluster_centers_는 중심
kmeans.labels_
kmeans.cluster_centers_

# 표시한 클러스터를 데이터프레임에 추가
df['cluster_id'] = kmeans.labels_

# x축, y축 설정, 출력할 데이터 설정, 선형회귀선, 점크기, hue파라미터로 categorial
sns.lmplot(x = "x", y = "y", data=df, fit_reg=False, scatter_kws={"s":150}, hue="cluster_id")
plt.title('after kmeans clustering')
plt.show()