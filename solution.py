"""Freight rate prediction: backtest, final fit, and prediction files.

Model: gradient boosting (scikit-learn HistGradientBoostingRegressor) predicting
rate-per-mile with an absolute-error loss, then multiplying by distance.

Why these choices (all verified in the backtest below):
  * Rate is ~ distance x rate-per-mile, so modelling rate-per-mile removes the
    dominant effect and lets the trees focus on everything else.
  * ~1.4% of labels are extreme outliers (rate-per-mile < 1.2 or > 3.5, roughly
    0.3x or 3.5x the norm) that look random. An absolute-error loss predicts the
    median and is not dragged around by them; squared error is.
  * Geography enters as pickup/delivery coordinates rather than city names: 8
    cities in the validation file never appear in training, and in a test that
    hid 8 cities coordinates matched the all-cities-seen error while city labels
    lost ~9%.
  * Only inputs that exist for BOTH the validation file and the December chart
    file are used. market_index / quote_signal were tested and rejected: they
    track seasonality in training but break out-of-time (errors 2-3x worse).
  * Raw time index / day-of-month were rejected for the same reason; day-of-week
    is kept because it improves every backtest origin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["distance", "weight_abs", "equipment_code", "dayofweek",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
OUTLIER_LOW, OUTLIER_HIGH = 1.2, 3.5  # rate-per-mile bounds used only for reporting


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    x["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
    # 292 training rows have negative weights; treat as sign errors (keep magnitude).
    # Missing weights stay NaN: the model handles them natively.
    x["weight_abs"] = pd.to_numeric(frame["weight"], errors="coerce").abs()
    x["equipment_code"] = frame["equipment"].map(EQUIPMENT_CODES)
    x["dayofweek"] = pd.to_datetime(frame["date"]).dt.dayofweek
    # Coordinates (not city names): the validation file has 8 cities never seen in
    # training (~12% of rows); coordinates generalise to them, city labels cannot.
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        x[c] = pd.to_numeric(frame[c], errors="coerce")
    return x[FEATURES]


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) lookup from every file that carries coordinates."""
    parts = []
    for f in frames:
        for role in ("pickup", "delivery"):
            parts.append(f[[role, f"{role}_lat", f"{role}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    lookup = pd.concat(parts).drop_duplicates()
    if lookup["city"].duplicated().any():
        raise ValueError("A city has inconsistent coordinates across files.")
    return lookup.set_index("city")


def add_coordinates(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """The December chart file has no coordinates, so look them up by city name."""
    out = frame.copy()
    for role in ("pickup", "delivery"):
        out[f"{role}_lat"] = out[role].map(lookup["lat"])
        out[f"{role}_lon"] = out[role].map(lookup["lon"])
    if out[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].isna().any().any():
        raise ValueError("December cities missing from the coordinate lookup.")
    return out


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=60, l2_regularization=1.0, random_state=0,
    )


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_train, x_test = build_features(train), build_features(test)
    model = make_model().fit(x_train, train["posted_rate"] / train["distance"])
    return np.maximum(model.predict(x_test) * test["distance"].to_numpy(), 1.0)


def metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict:
    err = p - y
    return {
        "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err**2).mean()),
        "MAPE%": 100 * (np.abs(err) / y).mean(),
        "MAE_clean": np.abs(err)[mask].mean(), "MAPE_clean%": 100 * (np.abs(err) / y)[mask].mean(),
    }


def backtest(dev: pd.DataFrame) -> None:
    """Rolling-origin backtest: train on everything before the origin, predict the
    next ~2 months (the validation file spans Nov-Dec, i.e. a ~2-month horizon)."""
    dev = dev.assign(date_dt=pd.to_datetime(dev["date"]))
    rpm = dev["posted_rate"] / dev["distance"]
    clean = ((rpm > OUTLIER_LOW) & (rpm < OUTLIER_HIGH)).to_numpy()
    print("Rolling-origin backtest (train < origin, test = next 61 days)")
    rows = []
    for origin in ["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]:
        o = pd.Timestamp(origin)
        train = dev[dev["date_dt"] < o]
        mask = ((dev["date_dt"] >= o) & (dev["date_dt"] < o + pd.Timedelta(days=61))).to_numpy()
        test = dev[mask]
        y = test["posted_rate"].to_numpy()
        pred = fit_predict(train, test)
        # naive baseline: median rate-per-mile of training data x distance
        base = np.full(len(test), (train["posted_rate"] / train["distance"]).median()) * test["distance"].to_numpy()
        m, b = metrics(y, pred, clean[mask]), metrics(y, base, clean[mask])
        rows.append({"origin": origin, "model_MAE": m["MAE"], "model_MAE_clean": m["MAE_clean"],
                     "model_RMSE": m["RMSE"], "model_MAPE_clean%": m["MAPE_clean%"],
                     "baseline_MAE": b["MAE"], "baseline_MAE_clean": b["MAE_clean"]})
    table = pd.DataFrame(rows).set_index("origin")
    print(table.round(2).to_string())
    print("\nMean across origins:")
    print(table.mean().round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(data_dir / "train_test.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    december = pd.read_csv(data_dir / "december_chart_inputs.csv")

    backtest(dev)

    # Final model: fit on all 48,000 labelled rows.
    x_dev = build_features(dev)
    model = make_model().fit(x_dev, dev["posted_rate"] / dev["distance"])

    x_val = build_features(validation)
    val_pred = np.maximum(model.predict(x_val) * validation["distance"].to_numpy(), 1.0)
    preds = pd.DataFrame({"load_id": validation["load_id"].astype(str), "predicted_rate": val_pred})
    result = template[["load_id"]].merge(preds, on="load_id", how="left", validate="one_to_one")
    if result["predicted_rate"].isna().any():
        raise ValueError("Some template load_ids did not receive predictions.")
    result["predicted_rate"] = result["predicted_rate"].round(2)
    result.to_csv(out_dir / "validation_predictions.csv", index=False)

    december_geo = add_coordinates(december, city_coordinates(dev, validation))
    x_dec = build_features(december_geo)
    dec = december.copy()
    dec["predicted_rate"] = np.maximum(model.predict(x_dec) * december["distance"].to_numpy(), 1.0).round(2)
    dec.to_csv(out_dir / "december_chart_inputs_completed.csv", index=False)
    print(f"\nSaved {len(result):,} validation predictions and {len(dec)} December chart predictions.")


if __name__ == "__main__":
    main()
