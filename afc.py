import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn.decomposition import PCA
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


fa = FactorAnalyzer(n_factors = 6, rotation=None)
fa.fit(data_scaled)
ev, v = fa.get_eigenvalues()
print(ev)

