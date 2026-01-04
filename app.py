# app_v5_dn_level.py
# DN-Level v5 (Descriptive) — Raw → T_level → E_level → vE_level
# UI: 1 input table, 1 button, 1 output table, 1 label

import io
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Core math ----------
def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def compute_t_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anchors (illustrative normalization points, NOT clinical thresholds):
      HR:  good=60, bad=120  (higher worse)
      RR:  good=12, bad=30   (higher worse)
      SpO2: good=100, bad=85 (lower worse)
    T_level is clipped to [0,1]
    """
    out = df.copy()

    # HR (higher is worse): T = (HR - 60)/60
    out["T_HR"] = clip01((out["HR"].to_numpy(dtype=float) - 60.0) / 60.0)

    # RR (higher is worse): T = (RR - 12)/18
    out["T_RR"] = clip01((out["RR"].to_numpy(dtype=float) - 12.0) / 18.0)

    # SpO2 (lower is worse): T = (100 - SpO2)/15
    out["T_SpO2"] = clip01((100.0 - out["SpO2"].to_numpy(dtype=float)) / 15.0)

    return out


def compute_e_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["E_HR"] = 1.0 - out["T_HR"] ** 2
    out["E_RR"] = 1.0 - out["T_RR"] ** 2
    out["E_SpO2"] = 1.0 - out["T_SpO2"] ** 2
    return out


def compute_vE(df: pd.DataFrame) -> pd.DataFrame:
    """
    vE(t_i) = (E_i - E_{i-1}) / (Time_i - Time_{i-1})
    If Time is evenly spaced, that's fine; otherwise we honor dt.
    First row vE = NaN.
    """
    out = df.copy()
    t = out["Time"].to_numpy(dtype=float)
    dt = np.diff(t)

    # Guard against zero/negative dt (shouldn't happen after sorting, but just in case)
    if np.any(dt <= 0):
        raise ValueError("Time must be strictly increasing after sorting (no duplicates).")

    for col in ["E_HR", "E_RR", "E_SpO2"]:
        e = out[col].to_numpy(dtype=float)
        ve = np.full_like(e, fill_value=np.nan, dtype=float)
        ve[1:] = np.diff(e) / dt
        out["v" + col] = ve  # vE_HR, vE_RR, vE_SpO2

    return out


def label_from_vE(df: pd.DataFrame, eps: float = 1e-6) -> str:
    """
    1 label, descriptive only.
    Uses majority sign across available vE values at the latest timepoint.
    - declining: majority negative
    - recovering: majority positive
    - stable: otherwise (near-zero / mixed)
    """
    last = df.iloc[-1]
    ves = np.array([last["vE_HR"], last["vE_RR"], last["vE_SpO2"]], dtype=float)
    ves = ves[~np.isnan(ves)]
    if ves.size == 0:
        return "Reserve dynamics: N/A (need ≥2 timepoints)"

    neg = np.sum(ves < -eps)
    pos = np.sum(ves > eps)

    if neg > pos:
        return "Reserve dynamics: declining"
    if pos > neg:
        return "Reserve dynamics: recovering"
    return "Reserve dynamics: stable"


# ---------- UI helpers ----------
DEFAULT_CSV = """Time,HR,RR,SpO2
-20,92,16,98
-10,98,18,97
0,104,20,97
"""


def load_input_table(raw_text: str) -> pd.DataFrame:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Input is empty.")
    df = pd.read_csv(io.StringIO(raw_text))

    required = {"Time", "HR", "RR", "SpO2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Required: Time, HR, RR, SpO2")

    # Coerce numeric
    for c in ["Time", "HR", "RR", "SpO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["Time", "HR", "RR", "SpO2"]].isna().any().any():
        bad_rows = df[df[["Time", "HR", "RR", "SpO2"]].isna().any(axis=1)]
        raise ValueError(f"Non-numeric or missing values found in rows:\n{bad_rows}")

    # Sort by Time
    df = df.sort_values("Time").reset_index(drop=True)

    # Must have at least 2 points for vE
    if len(df) < 2:
        raise ValueError("Need at least 2 timepoints to compute vE.")

    # Ensure strictly increasing time
    if df["Time"].duplicated().any():
        raise ValueError("Time contains duplicates. Please make Time unique.")
    if not (df["Time"].diff().dropna() > 0).all():
        raise ValueError("Time must be strictly increasing after sorting.")

    return df


# ---------- Streamlit app ----------
st.set_page_config(page_title="DN-Level v5 (Descriptive)", layout="centered")
st.title("DN-Level v5 (Descriptive)")
st.caption(
    "Input raw time series (Time, HR, RR, SpO₂) → Compute DN-Level reserve (E) and reserve velocity (vE). "
    "Descriptive only. No decision logic."
)

st.subheader("Input (raw time series)")
raw_text = st.text_area(
    "Paste CSV (columns: Time, HR, RR, SpO2). Time can be -20, -10, 0, 10, 20, 30… any length.",
    value=DEFAULT_CSV,
    height=160,
)

compute = st.button("Compute DN-Level", type="primary")

if compute:
    try:
        df_in = load_input_table(raw_text)

        df = compute_t_level(df_in)
        df = compute_e_level(df)
        df = compute_vE(df)

        # Output table (as requested: just E & vE; keep raw Time for reference)
        out_cols = [
            "Time",
            "E_HR", "vE_HR",
            "E_RR", "vE_RR",
            "E_SpO2", "vE_SpO2",
        ]
        df_out = df[out_cols].copy()

        # Pretty formatting (but keep real numbers)
        st.subheader("Output (DN-Level)")
        st.dataframe(df_out, use_container_width=True)

        st.subheader("Label")
        st.write(label_from_vE(df))

        st.caption(
            "Note: Anchors used are illustrative normalization points (not clinical thresholds). "
            "This app is descriptive and does not provide alerts or recommendations."
        )

    except Exception as e:
        st.error(str(e))
