"""
generate_data.py - Synthetic retail sales / inventory time series (supply chain).
Modeled on public retail demand datasets (e.g., Store Item Demand). Daily units sold
for several products with trend + weekly seasonality + annual seasonality + promo spikes,
so forecasting and inventory optimization produce meaningful results.
"""
import numpy as np, pandas as pd, pathlib
rng = np.random.default_rng(11)
BASE = pathlib.Path(__file__).resolve().parent

start = pd.Timestamp("2023-01-01")
days = 730
dates = pd.date_range(start, periods=days, freq="D")
products = [("SKU-01","Beverages",120,0.9),("SKU-02","Snacks",90,1.1),
            ("SKU-03","Household",60,0.7),("SKU-04","Personal Care",45,0.6),
            ("SKU-05","Frozen",75,1.3),("SKU-06","Bakery",50,1.0)]
dow_factor = np.array([0.95,0.9,0.95,1.0,1.15,1.35,1.2])  # Mon..Sun, weekend lift

rows=[]
for sku,cat,base,seas_amp in products:
    trend = np.linspace(0, base*0.15, days)               # gentle upward trend
    t = np.arange(days)
    annual = seas_amp*base*0.18*np.sin(2*np.pi*(t/365.25) - 1.0)  # yearly cycle
    weekly = base*(dow_factor[dates.dayofweek]-1.0)
    promo = np.zeros(days)
    promo_days = rng.choice(days, size=days//45, replace=False)   # occasional promos
    promo[promo_days] = base*rng.uniform(0.4,0.9,len(promo_days))
    noise = rng.normal(0, base*0.10, days)
    units = np.clip(np.round(base+trend+annual+weekly+promo+noise),0,None).astype(int)
    for d,u,pr in zip(dates,units,promo):
        rows.append((d.date().isoformat(),sku,cat,int(u),int(pr>0)))
df = pd.DataFrame(rows, columns=["date","sku","category","units_sold","promo"])
df.to_csv(BASE/"sales.csv", index=False)
print("rows:",len(df),"| skus:",df.sku.nunique(),"| date range:",df.date.min(),"->",df.date.max())
print(df.groupby("sku")["units_sold"].mean().round(1).to_dict())
