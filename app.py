import os
import glob
import random
import pickle
from math import prod
from itertools import product

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kolkata House Price Predictor",
                   page_icon="🏠", layout="centered")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    background-attachment: fixed;
}
.block-container {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 22px;
    padding: 2.2rem 2.5rem !important;
    margin-top: 2rem;
    max-width: 850px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}
.app-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffd166, #ff7e5f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.app-sub {
    text-align: center;
    color: #cfd8dc;
    margin-bottom: 1.5rem;
}
label, .stMarkdown p { color: #eceff1 !important; font-weight: 600; }

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: white;
    font-size: 1.15rem;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 0;
    margin-top: 1rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(255, 126, 95, 0.45);
    color: white;
}
.result-card {
    text-align: center;
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #062a22;
    border-radius: 16px;
    padding: 1.4rem;
    margin-top: 1.2rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
}
.result-card .label { font-size: 1rem; font-weight: 600; opacity: 0.85; }
.result-card .price { font-size: 2.4rem; font-weight: 800; }
.result-card .range { font-size: 0.95rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

data = pickle.load(open('house_price_model.pkl', 'rb'))
pipe, columns, meta = data['pipeline'], data['columns'], data['meta']

st.markdown('<div class="app-title">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Estimate property prices across Kolkata</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Created by SOUVIK NASKAR</div>', unsafe_allow_html=True)


def find_col(name):
    return next((c for c in columns if c.lower() == name), None)


loc_col = find_col('locality')
pin_col = find_col('pincode')


# ---- valid (locality, pincode) pairs taken from the CSV ----
@st.cache_data
def load_pairs(lc, pc):
    folder = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(folder, '*.csv'))
    d = pd.read_csv(files[0])
    d.columns = d.columns.str.strip()
    d = d[[lc, pc]].dropna()
    d[lc] = d[lc].astype(str).str.strip()
    d[pc] = d[pc].astype(int)
    return sorted(set(zip(d[lc], d[pc])))


pairs = []
if loc_col and pin_col:
    try:
        pairs = load_pairs(loc_col, pin_col)
    except Exception as e:
        st.warning(f"Could not read locality/pincode pairs from the CSV: {e}")
use_pairs = len(pairs) > 0

# show Pincode right after Locality so it updates correctly
display_cols = list(columns)
if use_pairs:
    display_cols.remove(pin_col)
    display_cols.insert(display_cols.index(loc_col) + 1, pin_col)

choices = {}
cols = st.columns(2)
for i, col in enumerate(display_cols):
    m = meta[col]
    label = col.replace('_', ' ').title()
    with cols[i % 2]:
        if use_pairs and col == loc_col:
            all_locs = sorted({l for l, p in pairs})
            choices[col] = st.selectbox(label, ['Any'] + all_locs)
        elif use_pairs and col == pin_col:
            sel_loc = choices[loc_col]
            pins = sorted({p for l, p in pairs if sel_loc == 'Any' or l == sel_loc})
            choices[col] = st.selectbox(label, ['Any'] + pins, key=f"pin_{sel_loc}")
        elif m['type'] == 'cat' or m['options'] is not None:
            choices[col] = st.selectbox(label, ['Any'] + m['options'])
        elif m['is_int']:
            choices[col] = st.number_input(label, min_value=int(m['min']),
                                           max_value=int(m['max']), value=int(m['median']))
        else:
            choices[col] = st.number_input(label, min_value=m['min'],
                                           max_value=m['max'], value=m['median'])


def all_values(col):
    return meta[col]['options']


if st.button('Predict Price'):
    pair_cols = [loc_col, pin_col] if use_pairs else []
    other_cols = [c for c in columns if c not in pair_cols]
    value_lists = [all_values(c) if choices[c] == 'Any' else [choices[c]] for c in other_cols]

    if use_pairs:
        pair_rows = [(l, p) for l, p in pairs
                     if (choices[loc_col] == 'Any' or l == choices[loc_col])
                     and (choices[pin_col] == 'Any' or p == choices[pin_col])]
        value_lists.append(pair_rows)  # locality + pincode always stay matched

    total = prod(len(v) for v in value_lists)

    if total <= 5000:
        combos = list(product(*value_lists))
    else:
        combos = [tuple(random.choice(v) for v in value_lists) for _ in range(5000)]

    if use_pairs:
        rows = [c[:-1] + c[-1] for c in combos]  # flatten the (locality, pincode) pair
        df_in = pd.DataFrame(rows, columns=other_cols + pair_cols)[columns]
    else:
        df_in = pd.DataFrame(combos, columns=columns)

    prices = pipe.predict(df_in)

    if total == 1:
        st.markdown(
            f'<div class="result-card"><div class="label">Predicted Price</div>'
            f'<div class="price">₹ {prices[0]:.2f} Lakh</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result-card"><div class="label">Average Price</div>'
            f'<div class="price">₹ {prices.mean():.2f} Lakh</div>'
            f'<div class="range">Range: ₹ {prices.min():.2f} to ₹ {prices.max():.2f} Lakh '
            f'(based on {total:,} combinations)</div></div>',
            unsafe_allow_html=True)