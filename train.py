import os
import glob, pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.csv'))
if not files:
    raise SystemExit('No CSV found in the Dataset folder')
print('Using file:', files[0])

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip()

price_col = next((c for c in df.columns if 'price' in c.lower()), None)
if price_col is None:
    raise SystemExit(f'No price column found. Columns: {df.columns.tolist()}')

df = df.dropna(subset=[price_col])
y = df[price_col]
X = df.drop(columns=[price_col])
X = X.drop(columns=[c for c in X.columns
                    if c.lower() in ('id', 'index', 'unnamed: 0')])

for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)

cat_cols = X.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(str).str.strip()
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ]), cat_cols),
])

pipe = Pipeline([
    ('pre', pre),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
print('R2 score on test data:', round(r2_score(y_test, pipe.predict(X_test)), 3))

pipe.fit(X, y)  # final fit on all data

meta = {}
for c in X.columns:
    if c in cat_cols:
        meta[c] = {'type': 'cat', 'options': sorted(X[c].unique().tolist())}
    else:
        s = X[c].dropna()
        is_int = bool((s % 1 == 0).all())
        uniq = sorted(s.unique().tolist())
        meta[c] = {
            'type': 'num', 'is_int': is_int,
            'min': float(s.min()), 'max': float(s.max()),
            'median': float(s.median()),
            'options': [int(u) if is_int else float(u) for u in uniq] if len(uniq) <= 10 else None,
        }

# ---- locality -> pincode / latitude / longitude lookup ----
def find_col(name):
    return next((c for c in X.columns if c.lower() == name), None)

loc_col, pin_col, lat_col, lon_col = (find_col(n) for n in
                                      ('locality', 'pincode', 'latitude', 'longitude'))

loc_info = {}
if all([loc_col, pin_col, lat_col, lon_col]):
    for loc, g in X.groupby(loc_col):
        pin = g[pin_col].mode()[0]
        loc_info[loc] = {
            'pincode': pin.item() if hasattr(pin, 'item') else pin,
            'lat': float(g[lat_col].mean()),
            'lon': float(g[lon_col].mean()),
        }

pickle.dump({'pipeline': pipe, 'columns': X.columns.tolist(), 'meta': meta,
             'loc_info': loc_info},
            open('house_price_model.pkl', 'wb'))
print('Saved house_price_model.pkl with columns:', X.columns.tolist())