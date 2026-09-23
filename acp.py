# Script qui fait une analyse ACP des données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("steam.csv")

# 3. Analyse en composante principale
# On utilise les attributs suivants :
# price, positives_reviews, negative_reviews, average_playtime, median_playtime, achivements, required_age
colonnes_quantitatives = [
    'price', 
    'positive_ratings', 
    'negative_ratings', 
    'average_playtime', 
    'median_playtime', 
    'achievements', 
    'required_age'
]

# Standardisation des données
x = data[colonnes_quantitatives]

temp = x.sub(x.mean())
x_scaled = temp.div(x.std())


# On calcule la modélisation de l'ACP
pca = PCA(n_components=6)
pca_res = pca.fit_transform(x_scaled)

print(pca_res)

eig = pd.DataFrame({
    "Dimension":
    ["Dim" + str(x+1) for x in range(6)],
    "Valeur propre": pca.explained_variance_,
    "% valeur propre":
    np.round(pca.explained_variance_ratio_ * 100),
    "cum. valeur propre":
    np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
})

print(eig)

y1 = list(pca.explained_variance_ratio_)
x1 = range(len(y1))
plt.bar(x1,y1)
plt.show()