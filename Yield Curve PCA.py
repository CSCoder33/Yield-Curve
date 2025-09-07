import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


try:
    # Force a non-interactive backend for headless environments (CI)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def ensure_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)


# --- Data Loading ---

FRED_SERIES = {
    # maturity (years): FRED constant maturity series code
    1 / 12: "DGS1MO",   # 1 month
    3 / 12: "DGS3MO",   # 3 month
    6 / 12: "DGS6MO",   # 6 month
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30",
}


def _try_fetch_from_fred(start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch FRED yields.
    Strategy:
      1) Try pandas_datareader (if compatible)
      2) Fallback to direct CSV download from fredgraph (no API key)
    Returns DataFrame with columns as float maturities (years) in % yields.
    """
    # 1) pandas_datareader path
    try:
        from pandas_datareader import data as pdr  # type: ignore
        try:
            df_list = []
            for mat, code in FRED_SERIES.items():
                s = pdr.DataReader(code, "fred", start=start, end=end)
                s.rename(columns={code: mat}, inplace=True)
                df_list.append(s)
            df = pd.concat(df_list, axis=1).sort_index()
            return df
        except Exception:
            pass
    except Exception:
        pass

    # 2) Direct CSV download from fredgraph
    try:
        import io
        import requests
        params_common = {}
        if start:
            params_common["cosd"] = start
        if end:
            params_common["coed"] = end
        frames = []
        for mat, code in FRED_SERIES.items():
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
            params = {"id": code, **params_common}
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200 or not r.text:
                return None
            df = pd.read_csv(io.StringIO(r.text))
            # Normalize: FRED uses either 'DATE' or 'observation_date'
            date_col = None
            for cand in ("DATE", "observation_date", "date", "Date"):
                if cand in df.columns:
                    date_col = cand
                    break
            if date_col is None:
                return None
            df.rename(columns={date_col: "Date", code: str(mat)}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]) 
            # Convert to numeric, coerce missing '.' to NaN
            df[str(mat)] = pd.to_numeric(df[str(mat)], errors="coerce")
            df.set_index("Date", inplace=True)
            frames.append(df[[str(mat)]])
        out = pd.concat(frames, axis=1).sort_index()
        # Columns as float maturities
        out.columns = [float(c) for c in out.columns]
        return out
    except Exception:
        return None


def _try_read_local_csv(path: str = "data/treasury_yields.csv") -> Optional[pd.DataFrame]:
    """Read a local CSV if present. Accepts several common schemas:
    - Columns as FRED codes (DGS2, DGS10, ...)
    - Columns as maturities (e.g., 0.25, 2, 10) or labels (e.g., '2Y','10Y','1 Mo')
    Returns DataFrame with float maturity columns in % yields.
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    # Find date column
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "DATE".lower()) or "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        # Try index as date
        pass
    else:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    # Normalize columns to maturities
    col_map: Dict[str, float] = {}
    # First pass: FRED codes
    inv = {v: k for k, v in FRED_SERIES.items()}
    for c in df.columns:
        cc = c.strip().upper()
        if cc in inv:
            col_map[c] = float(inv[cc])

    # Second pass: common labels like '1 MO', '3M', '2Y', '10 YR'
    def parse_maturity_label(cc: str) -> Optional[float]:
        t = cc.replace(" ", "")
        # Examples: 1MO, 3M, 6M, 1Y, 2YR, 10Y, 30YR
        if t.endswith("MO") or t.endswith("M"):
            try:
                n = float(t[:-2] if t.endswith("MO") else t[:-1])
                return n / 12.0
            except Exception:
                return None
        if t.endswith("YR") or t.endswith("Y"):
            try:
                n = float(t[:-2] if t.endswith("YR") else t[:-1])
                return n
            except Exception:
                return None
        return None

    for c in df.columns:
        if c in col_map:
            continue
        cc = c.strip().upper()
        m = parse_maturity_label(cc)
        if m is not None:
            col_map[c] = float(m)

    # Third pass: numeric columns already
    for c in df.columns:
        if c in col_map:
            continue
        try:
            m = float(c)
            col_map[c] = m
        except Exception:
            continue

    # Build standardized DataFrame
    selected = {m: df[c] for c, m in col_map.items()}
    if not selected:
        return None
    out = pd.concat(selected, axis=1)
    # Ensure float columns as maturities
    out.columns = [float(c) for c in out.columns]
    out.sort_index(axis=1, inplace=True)
    out.sort_index(inplace=True)
    return out


def load_treasury_yields(
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_update_local: bool = True,
) -> pd.DataFrame:
    """Load constant maturity Treasury yields into a wide DataFrame (columns=maturities in years).
    Values are in percent. If a local CSV exists, use it; if it's stale and network is
    available, attempt to fetch missing dates from FRED and append, writing back to CSV.
    Otherwise, fall back to fetching the full history from FRED.
    """
    ensure_dirs()

    # 1) Try local CSV
    df_local = _try_read_local_csv()
    if df_local is not None:
        if auto_update_local:
            # If stale, try to update with FRED and write back
            try:
                last_local = df_local.index.max()
                # If end not provided, try through today
                tgt_end = pd.to_datetime(end) if end is not None else pd.Timestamp.today().normalize()
                # If we are missing at least one business day, try to fetch
                if pd.isna(last_local) or (last_local.date() < (tgt_end.date())):
                    fetch_start = (last_local + pd.Timedelta(days=1)).date() if pd.notna(last_local) else None
                    df_new = _try_fetch_from_fred(start=str(fetch_start) if fetch_start else start, end=end)
                    if df_new is not None and not df_new.empty:
                        # Standardize columns as maturities to match local
                        df_new.columns = [float(c) for c in df_new.columns]
                        # Align and append only newer rows
                        df_upd = pd.concat([df_local, df_new], axis=0)
                        df_upd = df_upd[sorted(set(df_upd.columns) | set(FRED_SERIES.keys()))]
                        df_upd = df_upd[sorted(df_upd.columns)]
                        df_upd = df_upd[~df_upd.index.duplicated(keep="last")].sort_index()
                        # Write back to CSV in the original labeled format
                        # Recreate a friendly header using FRED codes where available
                        inv_map = {k: v for k, v in FRED_SERIES.items()}
                        cols = []
                        for c in df_upd.columns:
                            cols.append(inv_map.get(float(c), str(c)))
                        tmp = df_upd.copy()
                        tmp.columns = cols
                        out_path = os.path.join("data", "treasury_yields.csv")
                        # Save with Date index label
                        tmp.to_csv(out_path, index_label="Date")
                        df_local = df_upd
            except Exception:
                # Non-fatal; just use local as-is
                pass
        return df_local

    # 2) FRED via pandas_datareader
    df_fred = _try_fetch_from_fred(start=start, end=end)
    if df_fred is not None:
        # Save a copy locally for next runs
        try:
            ensure_dirs()
            tmp = df_fred.copy()
            # Convert column names to FRED codes for readability
            cols = [FRED_SERIES.get(float(c), str(c)) for c in tmp.columns]
            tmp.columns = cols
            tmp.to_csv(os.path.join("data", "treasury_yields.csv"), index_label="Date")
        except Exception:
            pass
        return df_fred

    # 3) No data available
    raise FileNotFoundError(
        "No treasury data found. Place a CSV at 'data/treasury_yields.csv' with columns as FRED codes (e.g., DGS2, DGS10) or maturities (e.g., 0.25,2,10)."
    )


# --- Nelson–Siegel fitting (no SciPy) ---

@dataclass
class NSParams:
    beta0: float
    beta1: float
    beta2: float
    tau: float


def ns_basis(m: np.ndarray, tau: float) -> np.ndarray:
    # m: maturities in years (N,)
    x = m / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        L1 = (1 - np.exp(-x)) / x  # handle x=0 via lim -> 1
        L1 = np.where(np.isfinite(L1), L1, 1.0)
        L2 = L1 - np.exp(-x)
    # Return design matrix columns for beta0, beta1, beta2
    # y(m) = b0*1 + b1*L1 + b2*L2
    return np.vstack([np.ones_like(m), L1, L2]).T


def fit_ns_single(yields: pd.Series, tau_grid: Optional[np.ndarray] = None) -> Optional[NSParams]:
    """Fit Nelson–Siegel on a single date using grid search over tau and linear LS for betas.
    yields: Series indexed by maturity (years), values in decimal (not %).
    Returns NSParams or None if insufficient data.
    """
    s = yields.dropna()
    if s.size < 4:
        return None
    m = s.index.values.astype(float)
    y = s.values.astype(float)

    if tau_grid is None:
        tau_grid = np.linspace(0.25, 6.0, 40)

    best: Optional[Tuple[float, np.ndarray]] = None  # (rmse, [b0,b1,b2,tau])

    for tau in tau_grid:
        X = ns_basis(m, tau)
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        y_hat = X @ beta
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        if (best is None) or (rmse < best[0]):
            best = (rmse, np.array([beta[0], beta[1], beta[2], tau]))

    if best is None:
        return None
    b0, b1, b2, tau = best[1]
    return NSParams(float(b0), float(b1), float(b2), float(tau))


def ns_yield(m: np.ndarray, p: NSParams) -> np.ndarray:
    X = ns_basis(m, p.tau)
    beta = np.array([p.beta0, p.beta1, p.beta2])
    return X @ beta


# --- Zero/Forward Curves ---

def build_zero_curve(params: NSParams, grid: np.ndarray) -> pd.Series:
    """Return continuously compounded zero yields y(t) for grid (years) as decimals."""
    z = ns_yield(grid, params)
    return pd.Series(z, index=grid)


def build_discount_curve(zero_curve: pd.Series) -> pd.Series:
    """Compute discount factors P(t) = exp(-y(t)*t) from zero curve."""
    return np.exp(-zero_curve.values * zero_curve.index.values)


def build_forward_curve(zero_curve: pd.Series) -> pd.Series:
    """Approximate instantaneous forward rate f(t) = -d ln P(t)/dt via central difference."""
    t = zero_curve.index.values
    y = zero_curve.values
    P = np.exp(-y * t)
    # numerical derivative of -ln P(t)
    lnP = -np.log(P)
    f = np.zeros_like(lnP)
    # central differences; forward/backward at ends
    for i in range(len(t)):
        if 0 < i < len(t) - 1:
            dt = t[i + 1] - t[i - 1]
            f[i] = (lnP[i + 1] - lnP[i - 1]) / dt
        elif i == 0:
            dt = t[i + 1] - t[i]
            f[i] = (lnP[i + 1] - lnP[i]) / dt
        else:
            dt = t[i] - t[i - 1]
            f[i] = (lnP[i] - lnP[i - 1]) / dt
    return pd.Series(f, index=t)


# --- Bond Pricing & Risk ---

def price_bond_zero(face: float, coupon_rate: float, maturity: float, freq: int, zero_fn, shift: float = 0.0) -> float:
    """Price a bond using a supplied zero curve function y(t) (decimal, cont comp). Parallel shift in decimals.
    zero_fn: callable t-> y(t)
    shift: add to y(t) for all t (e.g., 0.0001 for +1bp)
    """
    # cash flow times
    n = int(round(maturity * freq))
    times = np.arange(1, n + 1, dtype=float) / freq
    c = face * coupon_rate / freq
    pv_cfs = []
    for t in times:
        y = zero_fn(t) + shift
        df = np.exp(-y * t)
        cf = c
        if abs(t - maturity) < 1e-9:
            cf += face
        pv_cfs.append(cf * df)
    return float(np.sum(pv_cfs))


def effective_duration_convexity(face: float, coupon_rate: float, maturity: float, freq: int, zero_fn, bump_bp: float = 1.0) -> Tuple[float, float]:
    """Effective duration and convexity via +/- parallel shift (bump in bps)."""
    b = bump_bp / 10000.0
    P0 = price_bond_zero(face, coupon_rate, maturity, freq, zero_fn, 0.0)
    P_up = price_bond_zero(face, coupon_rate, maturity, freq, zero_fn, +b)
    P_dn = price_bond_zero(face, coupon_rate, maturity, freq, zero_fn, -b)
    # Effective duration (per 1 change in yield), but we use b in decimals
    dur = (P_dn - P_up) / (2 * P0 * b)
    # Effective convexity
    conv = (P_up + P_dn - 2 * P0) / (P0 * (b ** 2))
    return float(dur), float(conv)


# --- PCA ---

@dataclass
class PCAResult:
    loadings: pd.DataFrame  # columns PC1.., index maturities
    explained_var: np.ndarray
    explained_ratio: np.ndarray


def run_pca(yields_wide: pd.DataFrame, n_components: int = 3) -> PCAResult:
    """Run PCA on de-meaned yields. Columns = maturities (years)."""
    Y = yields_wide.copy().dropna(how="any")
    if Y.shape[0] < 10:
        raise ValueError("Not enough observations for PCA after dropping NaNs")
    # De-mean columns
    Yc = Y - Y.mean(axis=0)
    # Covariance
    C = np.cov(Yc.values, rowvar=False)
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Select components
    eigvals_sel = eigvals[:n_components]
    eigvecs_sel = eigvecs[:, :n_components]
    # Normalize sign to make PC1 loadings mostly positive for interpretability
    for j in range(eigvecs_sel.shape[1]):
        if np.sum(eigvecs_sel[:, j]) < 0:
            eigvecs_sel[:, j] *= -1
    loadings = pd.DataFrame(eigvecs_sel, index=Y.columns, columns=[f"PC{i+1}" for i in range(n_components)])
    explained_ratio = eigvals_sel / eigvals.sum()
    return PCAResult(loadings=loadings, explained_var=eigvals_sel, explained_ratio=explained_ratio)


# --- Plotting helpers ---

def plot_curves(date_str: str, maturities: np.ndarray, par_y: np.ndarray, zero_y: np.ndarray, fwd_y: np.ndarray, out_path: str) -> None:
    if not HAS_MPL:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(maturities, par_y * 100, "o", label="Par (obs)")
    plt.plot(maturities, zero_y * 100, "-", label="Zero (NS)")
    plt.plot(maturities, fwd_y * 100, "--", label="Forward (inst approx)")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Yield / Rate (%)")
    plt.title(f"Curves on {date_str}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pca_loadings(loadings: pd.DataFrame, explained_ratio: np.ndarray, out_path: str) -> None:
    if not HAS_MPL:
        return
    plt.figure(figsize=(8, 5))
    for i, col in enumerate(loadings.columns):
        plt.plot(loadings.index.values.astype(float), loadings[col].values, label=f"{col} ({explained_ratio[i]*100:.1f}% var)")
    plt.axhline(0, color="k", lw=0.8, alpha=0.5)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Loading")
    plt.title("PCA Loadings (Level, Slope, Curvature)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --- Main orchestration ---

def main(start: Optional[str] = None, end: Optional[str] = None) -> None:
    ensure_dirs()

    try:
        y_wide = load_treasury_yields(start=start, end=end)
    except FileNotFoundError as e:
        print(str(e))
        print("Tip: If you want me to attempt FRED fetch, install pandas_datareader and ensure network, or save a CSV to data/treasury_yields.csv.")
        return

    # Keep a recent window for PCA speed (e.g., last 10 years)
    y_wide = y_wide.sort_index().dropna(how="all")
    # Standardize maturities subset present
    mats = sorted(set(y_wide.columns.astype(float)) & set(FRED_SERIES.keys()))
    y_wide = y_wide[mats]

    # Fill small gaps with forward fill on daily data
    y_wide = y_wide.ffill().dropna(how="any")

    # Convert to decimals for curves; keep a copy in % for display if needed
    y_dec = y_wide / 100.0

    # Choose last date with full data
    last_date = y_dec.index[-1]
    y_last = y_dec.loc[last_date]

    # Fit NS on last date
    ns = fit_ns_single(y_last)
    if ns is None:
        print("NS fit failed due to insufficient data on the last date.")
        return

    # Build curves on a dense grid
    grid = np.concatenate([
        np.linspace(1/12, 1.0, 12, endpoint=True),   # monthly to 1Y
        np.linspace(1.25, 5.0, 16, endpoint=True),
        np.linspace(6.0, 30.0, 25, endpoint=True),
    ])
    zero_curve = build_zero_curve(ns, grid)
    fwd_curve = build_forward_curve(zero_curve)

    # Plot curves vs observed par points
    if HAS_MPL:
        curves_path = os.path.join("reports/figures", f"curves_{last_date.date()}.png")
        curves_latest_path = os.path.join("reports/figures", "curves_latest.png")
        # Interpolate zero/fwd to observed maturities for an apples-to-apples plot
        par_mats = y_last.index.values.astype(float)
        zero_obs = np.interp(par_mats, zero_curve.index.values, zero_curve.values)
        fwd_obs = np.interp(par_mats, fwd_curve.index.values, fwd_curve.values)
        plot_curves(str(last_date.date()), par_mats, y_last.values, zero_obs, fwd_obs, curves_path)
        # Also write a stable filename for README embedding
        plot_curves(str(last_date.date()), par_mats, y_last.values, zero_obs, fwd_obs, curves_latest_path)

    # PCA on de-meaned yields (in %)
    try:
        pca_res = run_pca(y_wide.tail(252 * 10))  # ~10y of trading days if available
        if HAS_MPL:
            pca_path = os.path.join("reports/figures", "pca_loadings.png")
            plot_pca_loadings(pca_res.loadings, pca_res.explained_ratio, pca_path)
    except Exception as ex:
        print(f"PCA skipped: {ex}")
        pca_res = None

    # Bond ladder risk (equal-weight) using current curve
    # Ladder maturities and coupons set to current par yields at those maturities (approx par pricing)
    ladder_mats = [2.0, 5.0, 10.0, 30.0]
    available_mats = y_last.index.values.astype(float)
    coupons = {}
    for m in ladder_mats:
        # nearest available par yield for coupon
        idx = (np.abs(available_mats - m)).argmin()
        coupons[m] = float(y_last.iloc[idx])

    # Zero function via interpolation on our dense curve
    def zero_fn(t: float) -> float:
        return float(np.interp(t, zero_curve.index.values, zero_curve.values))

    items = []
    for m in ladder_mats:
        dur, conv = effective_duration_convexity(face=100.0, coupon_rate=coupons[m], maturity=m, freq=2, zero_fn=zero_fn, bump_bp=1.0)
        price = price_bond_zero(face=100.0, coupon_rate=coupons[m], maturity=m, freq=2, zero_fn=zero_fn, shift=0.0)
        items.append({"maturity": m, "coupon": coupons[m] * 100, "price": price, "duration": dur, "convexity": conv})

    ladder_df = pd.DataFrame(items).set_index("maturity").sort_index()
    port_weights = np.full(len(ladder_df), 1.0 / len(ladder_df))
    port_duration = float((ladder_df["duration"].values * port_weights).sum())
    port_convexity = float((ladder_df["convexity"].values * port_weights).sum())

    # Console summary
    print("=== Yield Curve & PCA Summary ===")
    print(f"Data dates: {y_wide.index[0].date()} -> {y_wide.index[-1].date()}  | Points per day: {len(mats)}")
    print(f"Last date: {last_date.date()}  NS params: beta0={ns.beta0:.4f}, beta1={ns.beta1:.4f}, beta2={ns.beta2:.4f}, tau={ns.tau:.3f}")
    if pca_res is not None:
        er = pca_res.explained_ratio
        print(f"PCA explained variance: PC1={er[0]*100:.1f}%, PC2={er[1]*100:.1f}%, PC3={er[2]*100:.1f}%")
    print("\nSample bond ladder (face=100, semiannual coupons):")
    print(ladder_df.assign(coupon=lambda d: d["coupon"].map(lambda x: f"{x:.2f}%")).rename(columns={"coupon": "coupon(% aa)", "price": "price($)"}))
    print(f"\nEqual-weight portfolio: Duration={port_duration:.2f}, Convexity={port_convexity:.2f}")

    if HAS_MPL:
        print("Figures saved:")
        print(f" - reports/figures/curves_{last_date.date()}.png")
        print(" - reports/figures/curves_latest.png")
        if pca_res is not None:
            print(" - reports/figures/pca_loadings.png")


if __name__ == "__main__":
    # Optional: set narrower window by passing start/end
    main()
