import pickle

with open('model/kmeans_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

print(f"Model Details:")
print(f"  - Number of clusters: {artifacts['optimal_k']}")
print(f"  - Model type: {type(artifacts['model'])}")
print(f"  - Scaler type: {type(artifacts['scaler'])}")
print(f"\nCluster Centers:")
print(artifacts['model'].cluster_centers_)

'''
Result:
Model Details:
  - Number of clusters: 4
  - Model type: <class 'sklearn.cluster._kmeans.KMeans'>
  - Scaler type: <class 'sklearn.preprocessing._data.MinMaxScaler'>

Cluster Centers:
[[0.04192812 0.01265023 0.07536138]
 [0.41797082 0.06300712 0.42553275]
 [0.21420173 0.02259228 0.22269012]
 [0.03809782 0.037792   0.28166302]]
'''