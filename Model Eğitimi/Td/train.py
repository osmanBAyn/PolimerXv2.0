import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

repo_id = "OsBaran/Polimer-Ozellik-Tahmini"
dataset = load_dataset(repo_id, split="Td")
df = pd.DataFrame(dataset)

time.sleep(5)
def smiles_to_fingerprint(smiles, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)
        return np.array(fp)
    except:
        return None

X_list = [smiles_to_fingerprint(s) for s in df['smiles']]
y_list = df['value'].tolist()

X = []
y = []
for i, fp in enumerate(X_list):
    if fp is not None:
        X.append(fp)
        y.append(y_list[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=1500, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=1500, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)
}


results_data = []
best_r2 = -np.inf
best_model_name = ""
best_y_pred = []


for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('regressor', model)])
    
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results_data.append({
        "Model": name,
        "RMSE": rmse,
        "R2 Score": r2
    })
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_y_pred = y_pred

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)

print("\n" + "="*40)
print(" MODELLERİN PERFORMANS TABLOSU")
print("="*40)
print(results_df)
print("="*40)

plt.figure(figsize=(8, 8))

plt.scatter(y_test, best_y_pred, alpha=0.6, color='blue', edgecolors='k', label='Veri Noktaları')

min_val = min(min(y_test), min(best_y_pred))
max_val = max(max(y_test), max(best_y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Mükemmel Tahmin (y=x)')

plt.title(f'En İyi Model Performansı: {best_model_name}\n($R^2$: {best_r2:.3f})', fontsize=24)
plt.xlabel('Gerçek Td Değerleri', fontsize=20)
plt.ylabel('Tahmin Edilen Td Değerleri', fontsize=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()