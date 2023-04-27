from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

##################
#   데이터 정리    #
##################
def init_data():
    # 붓꽃 데이터 가져오기
    # 가져와서 pandas 데이터프레임화
    # feature 항목을 열 이름으로(속성에 이름붙이기)
    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data)
    feature = pd.DataFrame(iris.feature_names)
    data.columns = feature[0]

    # 데이터마다 종류 할당
    # 컬럼명을 target으로
    target = pd.DataFrame(iris.target)
    target.columns = ['target']

    # 두 데이터프레임 합치기. axis = 1 -> 좌우로 합치기. 0이면 위아래
    df = pd.concat([data, target], axis=1)

    # df.info() 컬럼 타입 확인
    # 컬럼 타입 변경
    df = df.astype({'target': 'object'})

    # 데이터 요약 print(df.describe())

    return df

##################
#  수행전 시각화   #
##################

def visualization_seaborn(data):
    # hue = target을 기준으로 속성값 시각화
    # 인자로 전달되는 데이터프레임의 열을 두개씩 짝지을 수 있는 모든 조합에 대해 표현
    sns.pairplot(data, hue="target")
    plt.show()

def visualization_pyplot3d(data):
    # 클러스터링 이전 데이터셋 3d 시각화
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data.iloc[:, 0],
               data.iloc[:, 1],
               data.iloc[:, 2],
               s=10,
               cmap="orange",
               alpha=1,
               label='class1'
               )
    plt.legend()
    plt.show()


##################
#  K-Means 수행   #
##################

def find_cluster_opt(data):
    ks = range(1, 10)
    inertias = []

    # 적절한 클러스터 수 찾기
    for k in ks:
        model = KMeans(n_clusters=k, n_init='auto')
        model.fit(data)
        inertias.append(model.inertia_)

    plt.figure(figsize=(4, 4))
    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()

class clustering:
    def __init__(self):
        print("clustering")

    def Scailing_data(self, data):
        scailer = StandardScaler()
        scaled_data = pd.DataFrame(scailer.fit_transform(data.iloc[:,0:4]), columns = data.iloc[:,0:4].columns)
        target = pd.DataFrame(datasets.load_iris().target)
        target.columns = ['target']
        self.scaled_df = pd.concat([scaled_data, target], axis=1)
        self.non_scaled_df = data

    def K_Means(self, clusters = 3, algorithm = 'auto', max_iter = 1000, random_state = 42):
        clust_model = KMeans(n_clusters=clusters, n_init = algorithm, max_iter = max_iter, random_state = random_state)
        clust_model.fit(self.scaled_df)
        self.centers = clust_model.cluster_centers_
        self.pred = clust_model.predict(self.scaled_df)
        self.label = clust_model.labels_
        self.clust_df = self.scaled_df.copy()
        self.clust_df['clust_scaled'] = self.pred

        clust_model.fit(self.non_scaled_df)
        self.centers_non = clust_model.cluster_centers_
        self.pred_non = clust_model.predict(self.non_scaled_df)
        self.label = clust_model.labels_
        self.clust_df_non = self.non_scaled_df.copy()
        self.clust_df_non['clust'] = self.pred_non

    def visualization_2d(self):
        plt.figure(figsize=(20,6))
        plt.subplot(131)
        sns.scatterplot(x=self.clust_df.iloc[:,0],
                        y=self.clust_df.iloc[:,1],
                        data = self.clust_df,
                        hue = self.label,
                        palette = 'coolwarm')
        plt.scatter(self.centers[:,0],
                    self.centers[:,1],
                    c = 'black',
                    alpha = 0.8,
                    s = 150)


        plt.subplot(132)
        sns.scatterplot(x=self.clust_df.iloc[:, 0],
                        y=self.clust_df.iloc[:, 2],
                        data=self.clust_df,
                        hue=self.label,
                        palette='coolwarm')
        plt.scatter(self.centers[:, 0],
                    self.centers[:, 2],
                    c='black',
                    alpha=0.8,
                    s=150)


        plt.subplot(133)
        sns.scatterplot(x=self.clust_df.iloc[:, 0],
                        y=self.clust_df.iloc[:, 3],
                        data=self.clust_df,
                        hue=self.label,
                        palette='coolwarm')
        plt.scatter(self.centers[:, 0],
                    self.centers[:, 3],
                    c='black',
                    alpha=0.8,
                    s=150)

        plt.show()

    def visualization_3d(self):
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(self.clust_df.iloc[:,0],
                   self.clust_df.iloc[:,1],
                   self.clust_df.iloc[:,2],
                   c = self.clust_df.clust,
                   s =  10,
                   cmap = "rainbow",
                   alpha = 1
                   )
        ax.scatter(self.centers[:,0], self.centers[:,1], self.centers[:,2], c = 'black', s = 200, marker = '*')
        plt.show()

    def cluster_feat(self):
        print(pd.crosstab(self.clust_df['target'], self.clust_df['clust_scaled']))
        print(pd.crosstab(self.clust_df_non['target'], self.clust_df_non['clust']))