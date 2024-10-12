'Benötigte Libraries importieren'
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import make_blobs # "conda install scikit-learn" über Anaconda Prompt installieren
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler # kann weg vielleicht
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cluster, datasets, mixture

'Erstellen eines synthetischen 2D-Datensatzes'
n_samples = 300 # Anzahl der Datenpunkte
n_features = 2  # Anzahl der Features (Dimensionen)
n_clusters = 4  # Anzahl der Cluster

'Generiere Daten mit make_blobs'
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

"""
X ist ein NumPy-Array der Form (n_samples, n_features).
Jeder Eintrag in X enthält die Koordinaten eines Datenpunktes im Raum.
Y enthält die Labels (Clusterzugehörigkeit) für jeden Datenpunkt (0, 1, 2, ...)
und gibt an, zu welchem Cluster der entsprechende Punkt in X gehört.

make_blobs ist die Funktion, die die Daten erzeugt und enthält die zuvor definierten Parameter.

random_state=42 ist ein Zufalls-Seed, 42 ist willkürlich gewählt.
"""


# =============================================================================
# k-means Algorithmus implementieren
# =============================================================================

'k-means Algorithmus erstellen und trainieren'
kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
kmeans.fit(X)

"""
n_clusters=n_clusters legt Anzahl der Cluster fest, die der Algorithmus finden soll

Mit kmeans.fit(X) wird der k-means Algorithmus auf den Daten angewendet.
Dabei wird der Algorithmus iterativ die Cluster Zentroiden finden.
"""


'Vorhersagen (Cluster-Zuordnung für jeden Punkt)'
y_kmeans = kmeans.predict(X) # Array y_kmeans gibt für jeden Punkt das zugehörige Cluster an


'Visualisierung der Cluster'
plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=35, cmap="plasma")


"""
plt.scatter erzeugt Streudiagramm, um Datenpunkte im 2D-Raum darzustellen.
c=y_kmeans sorgt dafür, dass die Punkte in der Farbe des jeweiligen Clusters eingefärbt werden.

X[:,0] wählt erste Spalte des Arrays X aus, was die x-Koordinaten der Datenpunkte sind. 
(':' bedeutet 'alle Zeilen', '0' gibt erste Spalte des Arrays an)
X[:,1] wählt zweite Spalte des Arrays X aus, was die y-Koordinaten der Datenpunkte sind.

s steht für size (Größe) der Punkte
cmap="plasma" ist eine Farbpalette, die verwendet wird, um die Punkte basierend auf dem Wert 
von c einzufärben.
"""


centers = kmeans.cluster_centers_  # centers enthält die Koordinaten der berechneten Zentroiden
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=0.75, marker="X")
plt.title("k-means Clustering")
plt.xlabel("Features 1")
plt.ylabel("Features 2")
plt.show()

# =============================================================================
# DBSCAN Algorithmus implementieren
# =============================================================================

'DBSCAN-Modell erstellen und trainieren'
'eps ist der Radius, min_samples ist die minimale Anzahl von Punkten, um einen Cluster zu definieren'
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X) 

"""
fit_predict(X) wendet DBSCAN auf Daten an und gibt Clusterzuordnung für jeden
Punkt zurück.
Punkte, die als Noise betrachtet werden, werden mit Label -1 versehen
"""


'Visualisieren der DBSCAN-Cluster'
plt.scatter(X[:,0], X[:,1], c=y_dbscan, s=35, cmap="plasma") 

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# =============================================================================
# Silhouettenkoeffizient
# =============================================================================

"""
Messung, wie gut Datenpunkte zu ihren eigenen Clustern im Vergleich zu benachbarten
Clustern passen. Ein Wert nahe 1 zeigt an, dass der Punkt gut zu seinem Cluster
passt, während ein Wert nahe -1 anzeigt, dass der Punkt eher zu einem anderen Cluster 
gehört.
"""

silhouette_avg1 = silhouette_score(X, y_kmeans)
silhouette_avg2 = silhouette_score(X, y_dbscan)
print(f"Durchschnittlicher Silhouettenwert k-means: {silhouette_avg1}")
print(f"Durchschnittlicher Silhouettenwert DBSCAN: {silhouette_avg2}")

# =============================================================================
# Elbow-Methode für k-means
# =============================================================================

"""
WCSS ist eine Metrik, die misst, wie stark die Datenpunkte innerhalb eines 
Clusters um den Zentroid herum gruppiert sind. Sie berechnet die Summe der
quadrierten Abstände zwischen jedem Punkt und dem Zentroid seines Clusters.
Ein kleinerer WCCS-Wert bedeutet, dass die Datenpunkte gut zu ihrem Cluster passen.
"""


'Erstellung einer leeren Liste, um die WCSS-Werte für versch. Cluster-Anzahlen zu speichern'
wcss = [] 
for k in range(1,11):     
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
"""
for k in range(1,11): ist eine Schleife, die über die Werte von k (Anzahl der Cluster) 
von 1 bis 10 iteriert. Ziel ist es, für jede Cluster Anzahl den WCSS-Wert zu 
berechnen und zu speichern.

kmeans = KMeans(n_clusters=k, random state=42) bedeutet, dass in jeder Iteration 
der Schleife ein neues k-means Modell erstellt wird. n_clusters=k bedeutet,
dass der k-means Algorithmus k Cluster finden soll.  

kmeans.inertia_ gibt den WCSS-Wert für das aktuelle k-means Modell zurück
Die berechneten Werte werden mit append() zur Liste wcss hinzugefügt.
"""    
    
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow-Methode")
plt.xlabel("Anzahl der Cluster")
plt.ylabel("Within-Cluster Sum of Square (WCSS)")
plt.show()

# =============================================================================
# ILS-Methode: Quelle 3
# =============================================================================

# def ILS(df, labelColumn, outColumn="LS", iterative=True):
#     featureColumns = [i for i in df.columns if i != labelColumn]
#     indexNames = list(df.index.names)
#     oldIndex = df.index
#     df = df.reset_index(drop=False)
#     labelled = [group for group in df.groupby(df[labelColumn] !=0)
#                 ][True][1].fillna(0)
#     unlabelled = [group for group in df.groupby(df[labelColumn] !=0)
#                   ][False][1]
#     outD = []
#     outID = []
#     closeID = []
#     while len(unlabelled) > 0:
#         D = pairwise_distances(labelled[featureColumns].values,
#                                 unlabelled[featureColumns].values)
#         (posL, posUnL) = np.unravel_index(D.argmin(), D.shape)
#         idUnL = unlabelled.iloc[posUnL].name
#         idL = labelled.iloc[posL].name
#         unlabelled.loc[idUnL, labelColumn] = labelled.loc[idL,labelColumn]
#         # labelled = labelled.append(unlabelled.loc[idUnL])  # Code aus Datei
#         labelled = pd.concat([labelled, unlabelled.loc[[idUnL]]]) # ChatGPT Code

#         unlabelled.drop(idUnL, inplace = True)
#         outD.append(D.min())
#         outID.append(idUnL)
#         closeID.append(idL)
#     if len(labelled) + len(unlabelled) != len(df):
#         raise Exception(format(len(labelled), len(unlabelled),len(df)))
#     newIndex = oldIndex[outID]
#     orderLabelled = pd.Series(data=outD, index=newIndex, name="minR")
#     closest = pd.Series(data=closeID, index=newIndex,name="IDclosestLabel")
#     labelled = labelled.rename(columns={labelColumn : outColumn})
#     newLabels = labelled.set_index(indexNames)[outColumn]
#     return newLabels, pd.concat([orderLabelled,closest],axis = 1)



# def min_toCentroid(df, centroid=None, features=None):
#     if type(features) == type(None):
#         features = df.columns
#     if type(centroid) == type(None):
#         centroid = df[features].mean()
#     dist = df.apply(lambda row : sum([(row[j] - centroid[i])**2 for i, j in enumerate(features)]
#             ), axis = 1)
#     return dist.idxmin()


# def plot_ILSdistances(df, minR, centroid, label):
#     fig = plt.figure(figsize=(6,3))
#     fig.subplots_adjust(left=.07, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
#     ax = plt.subplot(1, 2, 1)
#     plt.ylim(0, 1)
#     plt.xticks(()); plt.yticks(())
#     ax.plot(range(len(minR)), minR, color="blue")
#     ax = plt.subplot(1, 2, 2)
#     plt.xticks(()); plt.yticks(())
#     plt.xlim(-3,3); plt.ylim(-3,3)
#     ax.scatter(df["x"].values, df["y"].values, s=4, color=colors[0])
#     ax.scatter(centroid[0], centroid[1], s=3, color="red",marker="x", linewidth = 20)
    
    
    
# def kMeans_success(df, k):
#     df["label"] = 0 
#     model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df[features])
#     df["kMean"] = model.labels_ + 1
#     fig = plt.figure(figsize=(3,3))
#     ax1 = plt.subplot(1,1,1)
#     plt.xticks(()); plt.yticks(())
#     ax1.scatter(df["x"].values, df["y"].values, s=4, color=colors[df["kMean"].values])
#     for label, group in df.groupby(by = "kMean"):
#         group = group.copy()
#         group["label"] = 0 
#         centroid = model.cluster_centers_[label-1]
#         group.loc[min_toCentroid(group[features]), "label"] = label
#         ti = time.time()
#         newL, orderedL = ILS(group, "label")
#         tf = time.time()
#         print('Iterative label spreading took {:.1f}s to label {} points'.format(
#             tf-ti, len(group)))
#         plot_ILSdistances(group, orderedL["minR"].values, centroid, label)


# N = 1500
# noisy_circles = datasets.make_circles(n_samples=N, factor=.5, noise=.05)
# noisy_moons = datasets.make_moons(n_samples=N, noise=.05)
# blobs = datasets.make_blobs(n_samples=N, random_state=8)
# no_structure = np.random.rand(N, 2), None
# RS = 170
# X, y = datasets.make_blobs(n_samples=N, random_state= RS)
# transformation = [[0.6, -0.6], [-0.4, 0.8]]
# X_aniso = np.dot(X, transformation)
# aniso = (X_aniso, y)
# varied = datasets.make_blobs(n_samples=N,
#                               cluster_std=[1.0, 2.5, 0.5],
#                               random_state=RS)
# ds = [noisy_circles, noisy_moons, varied, aniso, blobs, no_structure]
# features = ["x", "y"]
# X = []
# for i,j in enumerate(ds):
#     X.append(pd.DataFrame(StandardScaler().fit_transform(j[0])
#                 ,columns = features))
#     X[i].index.name = "ID"
# from itertools import cycle, islice
# colors = np.array(list(islice(cycle(['#837E7C', '#377eb8', '#ff7f00',
#                                       '4daf4a', '#f781bf', '#a65628',
#                                       '#984ea3', '#999999', '#e41a1c', '#dede00']
#                                       ),int(10))))

# """
# obiger Code ab N = 1500 ist doppelt im Dokument
# """





# def ILS_Single_Label(df, features = ["x", "y"]):
#     df = df.copy()
#     df["label"] = 0 
#     centroid = df[features].mean()
#     closestToCentroid = min_toCentroid(df[features],centroid=centroid,features=features) 
#     # obige Zeile (closestToCentroid) unvollständig in pdf
#     df.loc[closestToCentroid, "label"] = 1 # Chat GPT, muss vlt weg

#     df.loc[0, "label"] = 1 
#     newL, orderedL = ILS(df[features + ["label"]], "label")
#     plot_ILSdistances(df, orderedL["minR"].values, centroid, 1)
    

# ILS_Single_Label(X[4])

# =============================================================================
# ILS-Methode: Chat-GPT
# =============================================================================


labeled_indices = [0, 50, 100]  
unlabeled_indices = list(set(range(X.shape[0])) - set(labeled_indices))


labels = -np.ones(X.shape[0], dtype=int) 
labels[labeled_indices] = y_kmeans[labeled_indices]


dist_matrix = pairwise_distances(X)


r_min_values = []
order_of_labeling = []  


while np.any(labels == -1):  
    for i in unlabeled_indices:
        
        nearest_labeled_index = np.argmin([dist_matrix[i, j] for j in labeled_indices])
        nearest_label = labels[labeled_indices[nearest_labeled_index]]
        r_min = dist_matrix[i, labeled_indices[nearest_labeled_index]]  

       
        labels[i] = nearest_label
        
        
        labeled_indices.append(i)
        unlabeled_indices.remove(i)

       
        r_min_values.append(r_min)
        order_of_labeling.append(i)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  


ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
ax1.set_title("ILS Clustering")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")


ax2.plot(range(len(r_min_values)), r_min_values, marker='o', linestyle='-')
ax2.set_title("R_min Plot")
ax2.set_xlabel("Reihenfolge der Labelung")
ax2.set_ylabel("R_min (minimale Distanz)")


plt.tight_layout()
plt.show()



