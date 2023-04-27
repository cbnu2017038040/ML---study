import Kmeans_iris

def Select_Menu(Menu):
    if(Menu == 1):
        df = Kmeans_iris.init_data()
        kmeans_cluster1 = Kmeans_iris.clustering()
        kmeans_cluster1.Scailing_data(df)
        kmeans_cluster1.K_Means()
        kmeans_cluster1.cluster_feat()

    elif(Menu == 99):
        exit()

while(1):
    print("=========================================================")
    print("=  Select ML Algorith                                   =")
    print("=  1. K-Means                                           =")
    print("= 99. Quit                                              =")
    print("=========================================================")
    menu = int(input())
    Select_Menu(menu)