import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

data = pd.read_csv("steam.csv")

# 2 Analyse Factorielle des correspondances
data_crosstab = pd.crosstab(data['genres'], data['platforms'])

temp = data_crosstab.sub(data_crosstab.mean())
data_scaled = temp.div(data_crosstab.std())

print(data_crosstab)

chi_square_value, p_value = calculate_bartlett_sphericity(data_scaled)

print("Chi square : ", chi_square_value)
print("p : ", p_value)


# Première analyse sans rotation
fa = FactorAnalyzer(n_factors = 6, rotation=None)
fa.fit(data_scaled)
ev, v = fa.get_eigenvalues()
print(ev)

# Premier graphique
plt.scatter(range(1,data_scaled.shape[1]+1), ev)
plt.plot(range(1,data_scaled.shape[1]+1), ev)
plt.title("Scree Plot")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show()


# AFC avec 3 rotations différentes
methods = [
    ("FA No rotation", FactorAnalysis(2,)),
    ("FA Varimax", FactorAnalysis(2, rotation="varimax")),
    ("FA Quartimax", FactorAnalysis(2, rotation="quartimax")),
]
fig, axes = plt.subplots(ncols=3, figsize=(10, 8), sharex=True, sharey=True)

for ax, (method, fa) in zip(axes, methods):
    fa = fa.fit(data_scaled)

    components = fa.components_

    vmax = np.abs(components).max()
    ax.scatter(components[0 ,:], components[1, :])
    ax.axhline(0, -1, 1, color="k")
    ax.axvline(0, -1, 1, color="k")
    for i,j, z in zip(components[0, :], components[1, :], data_scaled.columns):
        ax.text(i+.02, j+.02, str(z), ha="center")
    for i,j, z in zip(components[0, :], components[1, :], data_scaled.index):
        ax.text(i+.02, j+.02, str(z), ha="center")
    ax.set_title(str(method))
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Factor 1")
    ax.set_xlabel("Factor 2")

plt.tight_layout()
plt.show()