import numpy as np
import pandas as pd

from src.config import DEFAULT_DURATION_MAP, INDEX_COLUMNS


def _normalize_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容常见债券指数列名写法，把其标准化为项目要求的字段名。"""
    rename_map = {
        "0-1Y": "0_1Y",
        "0_1Y": "0_1Y",
        "0-1年": "0_1Y",
        "0_1年": "0_1Y",
        "1-3Y": "1_3Y",
        "1_3Y": "1_3Y",
        "1-3年": "1_3Y",
        "1_3年": "1_3Y",
        "3-5Y": "3_5Y",
        "3_5Y": "3_5Y",
        "3-5年": "3_5Y",
        "3_5年": "3_5Y",
        "5-7Y": "5_7Y",
        "5_7Y": "5_7Y",
        "5-7年": "5_7Y",
        "5_7年": "5_7Y",
        "7-10Y": "7_10Y",
        "7_10Y": "7_10Y",
        "7-10年": "7_10Y",
        "7_10年": "7_10Y",
        "10-25Y": "10_25Y",
        "10_25Y": "10_25Y",
        "10-25年": "10_25Y",
        "10_25年": "10_25Y",
        "30Y": "30Y",
        "30年": "30Y",
    }

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={c: rename_map.get(c, c) for c in out.columns})

    missing = [c for c in INDEX_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"债券指数文件缺少列：{missing}，请检查列名是否为 {INDEX_COLUMNS}")

    return out


def _clean_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """清洗日期列中的非日期说明行（例如“数据来源：Wind”），并统一转换成 datetime。"""
    if "date" not in df.columns:
        return df

    out = df.copy()
    out["date"] = out["date"].astype(str).str.strip()

    valid_mask = out["date"].str.contains(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{4}/\d{1,2}/\d{1,2}$|^\d{8}$",
        regex=True,
        na=False,
    )
    out = out[valid_mask].copy()

    try:
        out["date"] = pd.to_datetime(out["date"], format="mixed", errors="coerce")
    except TypeError:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


def compute_daily_return(series: pd.Series) -> pd.Series:
    """基于序列的日收益率，适用于净值或指数。"""
    return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_leverage_ratio(series: pd.Series) -> pd.Series:
    """把披露杠杆率的百分数转换成比例，例如 110.05 -> 1.1005。"""
    s = pd.to_numeric(series, errors="coerce")
    if s.empty:
        return s
    mask = s > 2
    s = s.copy()
    s.loc[mask] = s.loc[mask] / 100.0
    return s


def load_uploaded_data(nav_file, index_file, report_file):
    """读取上传的三个 CSV 文件，并进行基本清理与类型转换。"""
    nav_df = pd.read_csv(nav_file)
    index_df = pd.read_csv(index_file)
    report_df = pd.read_csv(report_file)

    nav_df = _clean_date_column(nav_df)
    index_df = _clean_date_column(_normalize_index_columns(index_df))
    report_df = _clean_date_column(report_df)

    nav_df = nav_df[["fund_id", "date", "nav"]].dropna().sort_values(["fund_id", "date"]).reset_index(drop=True)
    index_df = index_df[["date"] + INDEX_COLUMNS].dropna().sort_values("date").reset_index(drop=True)

    report_df = report_df[["fund_id", "date", "leverage", "duration"]].dropna(subset=["fund_id", "date"]).sort_values(["fund_id", "date"]).reset_index(drop=True)
    report_df["leverage"] = _normalize_leverage_ratio(report_df["leverage"])
    report_df["duration"] = report_df["duration"].replace({0: np.nan})

    return nav_df, index_df, report_df


def prepare_fund_data(fund_id: str, nav_df: pd.DataFrame, index_df: pd.DataFrame, report_df: pd.DataFrame):
    """为单只基金构造日度样本，并将披露观测按最近有效交易日前向对齐到净值日期。"""
    fund_nav = nav_df[nav_df["fund_id"] == fund_id].copy()
    fund_nav["return"] = compute_daily_return(fund_nav["nav"])

    index_returns = index_df.copy()
    for col in INDEX_COLUMNS:
        index_returns[col] = compute_daily_return(index_returns[col])
    index_returns = index_returns.rename(columns={c: f"{c}_ret" for c in INDEX_COLUMNS})

    df = fund_nav.merge(index_returns, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    for col in [f"{c}_ret" for c in INDEX_COLUMNS]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fund_report = report_df[report_df["fund_id"] == fund_id].copy().sort_values("date")
    if not fund_report.empty:
        fund_nav_dates = df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
        aligned_report = pd.merge_asof(
            fund_nav_dates,
            fund_report[["date", "leverage", "duration"]].sort_values("date"),
            on="date",
            direction="backward",
        )
        df = df.merge(aligned_report, on="date", how="left")
    else:
        df["leverage"] = np.nan
        df["duration"] = np.nan

    df["leverage_obs"] = df["leverage"]
    df["duration_obs"] = df["duration"]

    return df


def get_initial_duration_for_state(report_df: pd.DataFrame, default_duration_map: dict):
    """为 EKF 初始状态提供一个合理的起始组合久期。"""
    valid = report_df["duration"].dropna()
    if not valid.empty:
        return float(valid.iloc[0])
    return float(np.median(list(default_duration_map.values())))
