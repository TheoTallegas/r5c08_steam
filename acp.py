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
pca = PCA(n_components=3)
pca_res = pca.fit_transform(x_scaled)

print(pca_res)