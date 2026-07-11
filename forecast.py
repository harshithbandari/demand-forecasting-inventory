"""
forecast.py - Demand Forecasting & Inventory Optimization
Per SKU: fit trend + weekly + monthly seasonality, backtest on a 60-day holdout (MAPE/RMSE),
forecast the next 30 days, then compute safety stock, reorder point, and order-up-to level
balancing stockout risk vs. excess inventory. Exports forecasts, an inventory plan, and charts.
"""
import pathlib, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(BASE/"sales.csv", parse_dates=["date"]).sort_values(["sku","date"])

HORIZON=30; HOLDOUT=60; LEAD=7; Z=1.65   # 95% service level

def seasonal_fit(train):
    t=np.arange(len(train)); y=train["units_sold"].values.astype(float)
    b,a=np.polyfit(t,y,1)                      # linear trend
    detr=y-(a+b*t)
    wk=pd.Series(detr).groupby(train["date"].dt.dayofweek.values).mean()
    mo=pd.Series(detr - wk.reindex(train["date"].dt.dayofweek.values).values)\
         .groupby(train["date"].dt.month.values).mean()
    return a,b,wk,mo,len(train)

def predict(a,b,wk,mo,dates,t0):
    dates=pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    t=np.arange(t0,t0+len(dates))
    base=a+b*t
    dow=dates.dt.dayofweek.values; month=dates.dt.month.values
    return np.clip(base + wk.reindex(dow).values + mo.reindex(month).values,0,None)

fc_rows=[]; inv_rows=[]; mapes={}
for sku,g in df.groupby("sku"):
    g=g.reset_index(drop=True)
    train,test=g.iloc[:-HOLDOUT],g.iloc[-HOLDOUT:]
    a,b,wk,mo,n=seasonal_fit(train)
    pred=predict(a,b,wk,mo,test["date"],len(train))
    actual=test["units_sold"].values
    mape=float(np.mean(np.abs((actual-pred)/np.clip(actual,1,None)))*100)
    rmse=float(np.sqrt(np.mean((actual-pred)**2)))
    mapes[sku]=round(mape,1)
    # refit on full series, forecast next HORIZON days
    a,b,wk,mo,n=seasonal_fit(g)
    fut=pd.date_range(g["date"].iloc[-1]+pd.Timedelta(days=1),periods=HORIZON,freq="D")
    fut_pred=predict(a,b,wk,mo,fut,len(g))
    for d,v in zip(fut,fut_pred): fc_rows.append((sku,d.date().isoformat(),round(float(v),1)))
    # inventory optimization
    daily_std=float(g["units_sold"].tail(90).std())
    mean_daily=float(fut_pred.mean())
    lt_demand=mean_daily*LEAD
    safety=Z*daily_std*np.sqrt(LEAD)
    reorder=lt_demand+safety
    order_up_to=mean_daily*(LEAD+HORIZON/HORIZON*7)+safety   # cover lead + review period
    inv_rows.append((sku,g["category"].iloc[0],round(mean_daily,1),round(daily_std,1),
                     round(lt_demand,0),round(safety,0),round(reorder,0),round(order_up_to,0)))

fc=pd.DataFrame(fc_rows,columns=["sku","date","forecast_units"])
inv=pd.DataFrame(inv_rows,columns=["sku","category","avg_daily_demand","demand_std",
     "lead_time_demand","safety_stock","reorder_point","order_up_to_level"])
fc.to_csv(BASE/"forecast_30d.csv",index=False)
inv.to_csv(BASE/"inventory_recommendations.csv",index=False)

# ---- charts ----
plt.rcParams.update({"figure.dpi":120,"font.size":10})
# forecast vs actual (total across SKUs)
tot=df.groupby("date")["units_sold"].sum()
a,b,wk,mo,n=seasonal_fit(df.groupby("date")["units_sold"].sum().reset_index().assign(sku="ALL"))
plt.figure(figsize=(8,3.2))
plt.plot(tot.index[-120:],tot.values[-120:],label="Actual",color="#34495e")
futd=pd.date_range(tot.index[-1]+pd.Timedelta(days=1),periods=HORIZON)
plt.plot(futd,predict(a,b,wk,mo,futd,len(tot)),"--",label="Forecast (30d)",color="#e67e22")
plt.title("Total daily demand: recent actual + 30-day forecast"); plt.legend(); plt.tight_layout()
plt.savefig(BASE/"forecast_vs_actual.png"); plt.close()
# weekly seasonality
wkf=df.assign(dow=df["date"].dt.day_name().str[:3]).groupby(df["date"].dt.dayofweek)["units_sold"].mean()
ax=wkf.plot(kind="bar",color="#2980b9"); ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],rotation=0)
ax.set_ylabel("Avg units"); ax.set_title("Weekly demand seasonality"); plt.tight_layout()
plt.savefig(BASE/"weekly_seasonality.png"); plt.close()
# MAPE by SKU
ax=pd.Series(mapes).plot(kind="bar",color="#16a085"); ax.set_ylabel("MAPE %")
ax.set_title("Backtest forecast error by SKU (lower is better)"); plt.tight_layout()
plt.savefig(BASE/"mape_by_sku.png"); plt.close()
# reorder point vs safety stock
axx=inv.set_index("sku")[["lead_time_demand","safety_stock"]].plot(kind="bar",stacked=True,
     color=["#3498db","#e74c3c"]); axx.set_ylabel("Units"); axx.set_title("Reorder point = lead-time demand + safety stock")
plt.tight_layout(); plt.savefig(BASE/"reorder_points.png"); plt.close()

results={"skus":int(df.sku.nunique()),"days":int(df.date.nunique()),
 "backtest_mape_by_sku":mapes,"avg_mape":round(float(np.mean(list(mapes.values()))),1),
 "forecast_horizon_days":HORIZON,"lead_time_days":LEAD,"service_level":"95%",
 "total_reorder_units":int(inv["reorder_point"].sum())}
(BASE/"results.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
print(inv.to_string(index=False))
