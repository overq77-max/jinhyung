"""
Preprocessing script for KYRBS adolescent suicide research.
All comments converted to English for public release.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


RAW_CSV = "kyrbs_merged.csv"
OUT_CSV = "analysis_variables_only.csv"


def compute_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute BMI numeric value and categorize BMI into Low/Middle/High.
    BMI_VALUE is retained for reference, BMI is the categorical version.
    """
    df = df.copy()
    df["HT"] = pd.to_numeric(df["HT"], errors="coerce") / 100.0
    df["WT"] = pd.to_numeric(df["WT"], errors="coerce")
    df["BMI_VALUE"] = df["WT"] / (df["HT"] ** 2)

    q5 = df["BMI_VALUE"].quantile(0.05)
    q85 = df["BMI_VALUE"].quantile(0.85)

    df["BMI"] = pd.cut(
        df["BMI_VALUE"],
        bins=[-float("inf"), q5, q85, float("inf")],
        labels=["Low", "Middle", "High"],
        right=False,  # include q5, exclude q85
    )
    return df


def compute_weekday_sleep_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekday sleep duration and classify into 3 sleep categories.
    Sleep duration is calculated in minutes between sleep time and wake time.
    If sleep spans past midnight, adjust by adding 1440 minutes.
    """
    df = df.copy()
    cols = ["M_WK_HR", "M_WK_MM", "M_SLP_HR", "M_SLP_MM"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sleep_min = df["M_SLP_HR"] * 60 + df["M_SLP_MM"]
    wake_min = df["M_WK_HR"] * 60 + df["M_WK_MM"]

    duration = wake_min - sleep_min
    duration = duration.where(duration >= 0, duration + 1440)
    df["WEEKDAY_SLEEP_DURATION"] = duration

    def categorize(x: float) -> float:
        if pd.isna(x):
            return np.nan
        if x < 240:
            return 1  # <4 hours
        elif x <= 480:
            return 2  # 4–8 hours
        else:
            return 3  # >8 hours

    df["WEEKDAY_SLEEP_CATEGORY"] = df["WEEKDAY_SLEEP_DURATION"].apply(categorize)
    return df


def recode_city_type(df: pd.DataFrame) -> pd.DataFrame:
    """Recode city type categories into numeric values."""
    df = df.copy()
    df["CTYPE"] = df["CTYPE"].map({"대도시": 1, "중소도시": 2, "군지역": 3})
    return df


def recode_suicidal_variables(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    KYRBS suicidal questions are originally coded as 1/2.
    Shift them by +1 to make them 2 = No, 3 = Yes.
    Later, model scripts map {2:0, 3:1}.
    """
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c] + 1
    return df


def main():
    """Main preprocessing pipeline."""
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")

    df = recode_city_type(df)
    df = compute_bmi_category(df)
    df = compute_weekday_sleep_category(df)

    suicidal_cols = ["M_SUI_CON", "M_SUI_PLAN", "M_SUI_ATT"]
    present = [c for c in suicidal_cols if c in df.columns]
    df = recode_suicidal_variables(df, present)

    # Feature selection based on analysis variables used in ML and network modeling
    keep_cols = sorted(
        set(
            [
                "YEAR", "W", "PSU_ID", "STRATA",
                "BMI", "BMI_VALUE",
                "WEEKDAY_SLEEP_DURATION", "WEEKDAY_SLEEP_CATEGORY",
                "CTYPE",
            ]
            + present
            + [
                "M_SAD", "M_STR", "M_LON", "M_SLP_EN", "V_TRT",
                "PR_HT", "PR_BI", "SEX",
                "AS_DG_LT", "RH_DG_LT", "ECZ_DG_LT",
                "PA_TOT", "PA_VIG_D", "PA_MSC",
                "AC_LT", "TC_LT", "DR_HAB_PUR", "S_SI",
                "F_BR", "F_FRUIT", "F_FASTFOOD", "F_CAFF_A", "F_WAT",
                "E_SES", "E_S_RCRD",
            ]
        )
        & set(df.columns)
    )

    df_out = df[keep_cols].copy()
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved] {OUT_CSV} (rows={len(df_out)}, cols={len(df_out.columns)})")


if __name__ == "__main__":
    main()
