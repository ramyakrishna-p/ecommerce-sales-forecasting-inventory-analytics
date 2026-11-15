import numpy as np, pandas as pd, json, pathlib
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(42)
root = pathlib.Path(__file__).parent
(root / "data").mkdir(exist_ok=True); (root / "results").mkdir(exist_ok=True)

# ---------- 1. Generate SYNTHETIC daily e-commerce sales (2 years) ----------
START = pd.Timestamp("2024-01-01"); DAYS = 730
dates = pd.date_range(START, periods=DAYS, freq="D")
# category: (base daily units, avg unit price GBP, annual growth, weekend uplift)
CATS = {
    "Electronics":     (120, 85.0, 0.35, 1.25),
    "Home & Kitchen":  (200, 32.0, 0.20, 1.30),
    "Clothing":        (260, 28.0, 0.10, 1.20),
    "Beauty":          (150, 18.0, 0.25, 1.10),
    "Sports":          (90,  45.0, 0.15, 1.45),
}
rows = []
for cat, (base, price, growth, wknd) in CATS.items():
    for i, d in enumerate(dates):
        trend = 1 + growth * (i / DAYS)                                  # steady growth trend
        weekly = wknd if d.dayofweek >= 5 else (0.82 if d.dayofweek == 0 else 1.0)  # weekly seasonality
        annual = 1 + 0.18 * np.sin(2 * np.pi * (d.dayofyear - 30) / 365.25)          # yearly seasonality
        promo = 1.6 if (d.month == 11 and d.day >= 24) or (d.month == 12 and d.day >= 18) else 1.0  # Black Friday / Christmas spike
        noise = np.random.lognormal(mean=0, sigma=0.16)
        units = max(0, int(round(base * trend * weekly * annual * promo * noise)))
        unit_price = round(price * float(np.random.normal(1, 0.03)), 2)
        orders = max(1, int(round(units / np.random.uniform(1.4, 2.2))))
        rows.append([d.date(), cat, orders, units, round(units * unit_price, 2)])
sales = pd.DataFrame(rows, columns=["date", "category", "orders", "units_sold", "revenue"])
sales["date"] = pd.to_datetime(sales["date"])
sales.to_csv(root / "data/daily_sales.csv", index=False)

# ---------- 2. Aggregate to daily / weekly revenue and units ----------
daily = sales.groupby("date").agg(daily_revenue=("revenue", "sum"), daily_units=("units_sold", "sum"),
                                  daily_orders=("orders", "sum")).reset_index()
weekly = (sales.assign(week=sales["date"].dt.to_period("W").apply(lambda p: p.start_time))
               .groupby("week").agg(weekly_revenue=("revenue", "sum"), weekly_units=("units_sold", "sum")).reset_index())
weekly.to_csv(root / "results/weekly_summary.csv", index=False)

# ---------- 3. Forecasting evaluation: time-based holdout (last 56 days) ----------
def make_features(df):
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek; df["month"] = df["date"].dt.month
    df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = df["daily_revenue"].shift(lag)
    df["roll_7"] = df["daily_revenue"].shift(1).rolling(7).mean()
    df["roll_28"] = df["daily_revenue"].shift(1).rolling(28).mean()
    return df.dropna().reset_index(drop=True)

FEATS = ["dow", "month", "doy_sin", "doy_cos", "lag_1", "lag_7", "lag_14", "roll_7", "roll_28"]
feat = make_features(daily)
HOLDOUT = 56
train, test = feat.iloc[:-HOLDOUT], feat.iloc[-HOLDOUT:]

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": round(float(np.mean(np.abs(y_true - y_pred))), 2),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 2),
            "MAPE_pct": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2)}

baseline_pred = test["lag_7"].values  # seasonal-naive: same weekday last week
model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.06, subsample=0.9, random_state=42)
model.fit(train[FEATS], train["daily_revenue"])
ml_pred = model.predict(test[FEATS])
holdout = {"seasonal_naive": metrics(test["daily_revenue"], baseline_pred),
           "gradient_boosting": metrics(test["daily_revenue"], ml_pred)}
pd.DataFrame(holdout).T.reset_index(names="model").to_csv(root / "results/forecast_metrics.csv", index=False)

# ---------- 4. Forward forecast: next 28 days (recursive, retrained on all data) ----------
full_model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.06, subsample=0.9, random_state=42)
full_model.fit(feat[FEATS], feat["daily_revenue"])
history = list(daily["daily_revenue"].values)
last_date = daily["date"].max()
future_rows = []
for h in range(1, 29):
    d = last_date + pd.Timedelta(days=h)
    s = pd.Series(history)
    f = {"dow": d.dayofweek, "month": d.month,
         "doy_sin": np.sin(2 * np.pi * d.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * d.dayofyear / 365.25),
         "lag_1": s.iloc[-1], "lag_7": s.iloc[-7], "lag_14": s.iloc[-14],
         "roll_7": s.iloc[-7:].mean(), "roll_28": s.iloc[-28:].mean()}
    pred = float(full_model.predict(pd.DataFrame([f])[FEATS])[0])
    history.append(pred)
    future_rows.append([d.date(), round(pred, 2)])
forecast = pd.DataFrame(future_rows, columns=["date", "forecast_revenue"])
forecast.to_csv(root / "results/forecast_next_28_days.csv", index=False)
forecast_total = round(float(forecast["forecast_revenue"].sum()), 2)

# ---------- 5. Inventory analytics per category (last 90 days of demand) ----------
LEAD_TIME_DAYS = 7; SERVICE_LEVEL = 0.95; Z = 1.65  # z for 95% cycle service level
recent = sales[sales["date"] > last_date - pd.Timedelta(days=90)]
inventory = []
for cat, g in recent.groupby("category"):
    daily_units = g.groupby("date")["units_sold"].sum()
    avg_demand = float(daily_units.mean()); std_demand = float(daily_units.std(ddof=1))
    ltd = avg_demand * LEAD_TIME_DAYS
    safety_stock = Z * std_demand * np.sqrt(LEAD_TIME_DAYS)
    inventory.append({"category": cat, "avg_daily_demand": round(avg_demand, 2),
                      "demand_std": round(std_demand, 2), "lead_time_days": LEAD_TIME_DAYS,
                      "lead_time_demand": round(ltd, 1), "safety_stock": int(round(safety_stock)),
                      "reorder_point": int(round(ltd + safety_stock))})
inventory_df = pd.DataFrame(inventory).sort_values("reorder_point", ascending=False)
inventory_df.to_csv(root / "results/inventory_policy.csv", index=False)

# ---------- 6. Save summary ----------
summary = {
    "data_note": "SYNTHETIC data generated by analysis.py (seed 42). No real sales data.",
    "dataset": {"days": int(DAYS), "categories": int(sales["category"].nunique()),
                "rows_day_category": int(len(sales)), "total_orders": int(sales["orders"].sum()),
                "total_units": int(sales["units_sold"].sum()), "total_revenue": round(float(sales["revenue"].sum()), 2),
                "date_range": [str(sales["date"].min().date()), str(sales["date"].max().date())]},
    "holdout_days": HOLDOUT,
    "holdout_metrics": holdout,
    "forecast_next_28_days_total_revenue": forecast_total,
    "inventory_policy": {"lead_time_days": LEAD_TIME_DAYS, "service_level": SERVICE_LEVEL, "z": Z,
                         "by_category": inventory},
}
(root / "results/summary.json").write_text(json.dumps(summary, indent=2))
txt = ["E-commerce Sales Forecasting & Inventory Analytics — summary (SYNTHETIC data, seed 42)", "",
       f"Dataset: {DAYS} days ({summary['dataset']['date_range'][0]} to {summary['dataset']['date_range'][1]}), "
       f"{summary['dataset']['categories']} categories, {summary['dataset']['rows_day_category']} day-category rows, "
       f"total revenue £{summary['dataset']['total_revenue']:,.2f}", "",
       f"Forecast holdout (last {HOLDOUT} days, one-step-ahead using observed lags):"]
for name, m in holdout.items():
    txt.append(f"  {name}: MAE £{m['MAE']:,.2f} | RMSE £{m['RMSE']:,.2f} | MAPE {m['MAPE_pct']:.2f}%")
txt += ["", f"Next 28 days forecast revenue (recursive): £{forecast_total:,.2f}", "",
        f"Inventory policy (lead time {LEAD_TIME_DAYS} days, {int(SERVICE_LEVEL*100)}% service level, z={Z}):",
        inventory_df.to_string(index=False)]
(root / "results/summary.txt").write_text("\n".join(txt) + "\n")
print(json.dumps(summary, indent=2))
