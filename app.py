from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    DEFAULT_DAILY_LEVERAGE_CAP,
    DEFAULT_DAILY_WEIGHT_CAP,
    DEFAULT_DURATION_MAP,
    DEFAULT_LEVERAGE_PROCESS_NOISE,
    DEFAULT_OBSERVATION_NOISE_MATRIX,
    DEFAULT_WEIGHT_PROCESS_NOISE,
)
from src.data_utils import load_uploaded_data, prepare_fund_data
from src.ekf_model import estimate_daily_fund_states

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_outputs(daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df, duration_disagreement_df=None):
    daily_leverage_df.to_csv(OUTPUT_DIR / "daily_leverage.csv", index=False)
    daily_allocation_df.to_csv(OUTPUT_DIR / "daily_allocation.csv", index=False)
    daily_duration_df.to_csv(OUTPUT_DIR / "daily_duration.csv", index=False)
    duration_median_df.to_csv(OUTPUT_DIR / "duration_median.csv", index=False)
    if duration_disagreement_df is not None:
        duration_disagreement_df.to_csv(OUTPUT_DIR / "duration_disagreement.csv", index=False)


def compute_duration_disagreement_df(daily_duration_df: pd.DataFrame) -> pd.DataFrame:
    if daily_duration_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "asset_duration_mean",
                "asset_duration_std",
                "asset_duration_disagreement",
                "nav_duration_mean",
                "nav_duration_std",
                "nav_duration_disagreement",
            ]
        )

    stats = (
        daily_duration_df.groupby("date", as_index=False)
        .agg(
            asset_duration_mean=("asset_duration", "mean"),
            asset_duration_std=("asset_duration", "std"),
            nav_duration_mean=("nav_duration", "mean"),
            nav_duration_std=("nav_duration", "std"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    stats["asset_duration_disagreement"] = np.divide(
        stats["asset_duration_std"].fillna(0.0),
        stats["asset_duration_mean"].replace(0, np.nan).fillna(1.0),
        out=np.zeros(len(stats), dtype=float),
        where=np.abs(stats["asset_duration_mean"]) > 1e-12,
    )
    stats["nav_duration_disagreement"] = np.divide(
        stats["nav_duration_std"].fillna(0.0),
        stats["nav_duration_mean"].replace(0, np.nan).fillna(1.0),
        out=np.zeros(len(stats), dtype=float),
        where=np.abs(stats["nav_duration_mean"]) > 1e-12,
    )
    return stats


def parse_observation_noise_matrix(raw_matrix) -> np.ndarray:
    arr = np.asarray(raw_matrix, dtype=float)
    if arr.size == 3:
        arr = np.diag(arr)
    if arr.shape != (3, 3):
        raise ValueError("观测噪声矩阵必须为 3x3 矩阵或长度为 3 的对角向量。")
    return arr


def process_all_funds(
    nav_df,
    index_df,
    report_df,
    duration_map: Dict[str, float] | None = None,
    leverage_process_noise: float = DEFAULT_LEVERAGE_PROCESS_NOISE,
    weight_process_noise: float = DEFAULT_WEIGHT_PROCESS_NOISE,
    observation_noise_matrix=None,
    leverage_daily_cap: float = DEFAULT_DAILY_LEVERAGE_CAP,
    weight_daily_cap: float = DEFAULT_DAILY_WEIGHT_CAP,
):
    if duration_map is None:
        duration_map = DEFAULT_DURATION_MAP.copy()
    if observation_noise_matrix is None:
        observation_noise_matrix = DEFAULT_OBSERVATION_NOISE_MATRIX.copy()

    funds = sorted(nav_df["fund_id"].unique().tolist())
    all_daily = []

    for fund_id in funds:
        fund_df = prepare_fund_data(fund_id, nav_df, index_df, report_df)
        fund_result = estimate_daily_fund_states(
            fund_df,
            duration_map,
            report_df,
            leverage_process_noise=leverage_process_noise,
            weight_process_noise=weight_process_noise,
            observation_noise_matrix=observation_noise_matrix,
            leverage_daily_cap=leverage_daily_cap,
            weight_daily_cap=weight_daily_cap,
        )
        all_daily.append(fund_result)

    if not all_daily:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_daily_df = pd.concat(all_daily, ignore_index=True)
    all_daily_df = all_daily_df.sort_values(["fund_id", "date"]).reset_index(drop=True)

    daily_leverage_df = all_daily_df[["fund_id", "date", "leverage"]].copy()
    alloc_columns = [c for c in all_daily_df.columns if c.endswith("_weight")]
    daily_allocation_df = all_daily_df[["fund_id", "date"] + alloc_columns].copy()
    daily_duration_df = all_daily_df[["fund_id", "date", "asset_duration", "nav_duration", "leverage"]].copy()
    duration_median_df = (
        all_daily_df.groupby("date", as_index=False)[["asset_duration", "nav_duration"]]
        .median()
        .rename(columns={"asset_duration": "asset_duration_median", "nav_duration": "nav_duration_median"})
    )
    duration_disagreement_df = compute_duration_disagreement_df(daily_duration_df)

    return daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df, duration_disagreement_df


def render_duration_config() -> Dict[str, float]:
    st.subheader("步骤2：参数设置 - 七档债券指数久期")
    cols = st.columns(2)
    duration_map = {}
    for idx, key in enumerate(DEFAULT_DURATION_MAP.keys()):
        col = cols[idx % 2]
        with col:
            duration_map[key] = st.number_input(
                f"{key} 久期（年）",
                min_value=0.0,
                max_value=100.0,
                value=float(DEFAULT_DURATION_MAP[key]),
                step=0.01,
                format="%.2f",
                key=f"duration_{key}",
            )
    return duration_map


def render_model_config() -> Tuple[float, float, float, float, np.ndarray]:
    st.subheader("步骤2：参数设置 - EKF 模型参数")

    leverage_process_noise = st.number_input(
        "杠杆率变化参数",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_LEVERAGE_PROCESS_NOISE),
        step=1e-6,
        format="%.6f",
    )
    weight_process_noise = st.number_input(
        "期限权重变化参数",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_WEIGHT_PROCESS_NOISE),
        step=1e-8,
        format="%.8f",
    )
    leverage_daily_cap = st.number_input(
        "每日杠杆率最大变化约束",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_DAILY_LEVERAGE_CAP),
        step=1e-4,
        format="%.4f",
    )
    weight_daily_cap = st.number_input(
        "每个期限权重最大变化约束",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_DAILY_WEIGHT_CAP),
        step=1e-4,
        format="%.4f",
    )

    default_obs_df = pd.DataFrame(
        DEFAULT_OBSERVATION_NOISE_MATRIX,
        index=["return", "leverage", "duration"],
        columns=["return", "leverage", "duration"],
    )
    st.caption("观测噪声矩阵（按 [return, leverage, duration] 顺序）")
    obs_df = st.data_editor(
        default_obs_df,
        num_rows="fixed",
        hide_index=False,
        use_container_width=True,
    )
    observation_noise_matrix = parse_observation_noise_matrix(obs_df.to_numpy())
    return (
        float(leverage_process_noise),
        float(weight_process_noise),
        float(leverage_daily_cap),
        float(weight_daily_cap),
        observation_noise_matrix,
    )


def render_results(daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df, duration_disagreement_df, funds):
    st.subheader("步骤4：分析结果呈现")

    if not funds:
        st.warning("当前没有可展示的基金数据。")
        return

    if "selected_fund" not in st.session_state:
        st.session_state.selected_fund = funds[0]
    elif st.session_state.selected_fund not in funds:
        st.session_state.selected_fund = funds[0]

    selected_fund = st.selectbox(
        "选择要展示的基金",
        options=funds,
        index=funds.index(st.session_state.selected_fund),
        key="selected_fund",
    )

    filtered_leverage = daily_leverage_df[daily_leverage_df["fund_id"] == selected_fund].copy()
    filtered_duration = daily_duration_df[daily_duration_df["fund_id"] == selected_fund].copy()
    filtered_alloc = daily_allocation_df[daily_allocation_df["fund_id"] == selected_fund].copy()

    mean_values = filtered_duration[["leverage", "asset_duration", "nav_duration"]].mean()
    summary = pd.DataFrame(
        [
            {
                "fund_id": selected_fund,
                "avg_leverage": float(mean_values["leverage"]),
                "avg_asset_duration": float(mean_values["asset_duration"]),
                "avg_nav_duration": float(mean_values["nav_duration"]),
            }
        ]
    )

    st.subheader("关键指标概览")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("当前基金", selected_fund)
    with c2:
        st.metric("平均杠杆率", round(float(summary["avg_leverage"].iloc[0]), 4) if not summary.empty else 0.0)
    with c3:
        st.metric("平均组合久期", round(float(summary["avg_asset_duration"].iloc[0]), 2) if not summary.empty else 0.0)

    st.dataframe(summary[["fund_id", "avg_leverage", "avg_asset_duration", "avg_nav_duration"]], use_container_width=True)

    st.subheader("1. 日度杠杆率")
    fig_leverage = px.line(
        filtered_leverage,
        x="date",
        y="leverage",
        markers=True,
        title=f"{selected_fund} 日度杠杆率",
    )
    fig_leverage.add_hline(1.0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_leverage, use_container_width=True)

    st.subheader("2. 组合久期与净值久期")
    duration_plot_df = filtered_duration.melt(
        id_vars=["fund_id", "date"],
        value_vars=["asset_duration", "nav_duration"],
        var_name="duration_type",
        value_name="duration",
    )
    fig_duration = px.line(
        duration_plot_df,
        x="date",
        y="duration",
        color="duration_type",
        markers=True,
        title=f"{selected_fund} 组合久期与净值久期",
        labels={"duration_type": "久期类型", "duration": "久期（年）"},
    )
    st.plotly_chart(fig_duration, use_container_width=True)

    st.subheader("3. 七档期限配置权重变化")
    area_df = filtered_alloc.melt(id_vars=["fund_id", "date"], var_name="bucket", value_name="weight")
    fig_area = px.area(
        area_df,
        x="date",
        y="weight",
        color="bucket",
        line_group="fund_id",
        title=f"{selected_fund} 七档期限配置权重变化",
    )
    st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("4. 全市场久期中位数")
    fig_median = px.line(
        duration_median_df,
        x="date",
        y=["asset_duration_median", "nav_duration_median"],
        markers=True,
        title="全市场久期中位数趋势",
    )
    st.plotly_chart(fig_median, use_container_width=True)
    median_csv = duration_median_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="导出全市场久期中位数 CSV",
        data=median_csv,
        file_name="duration_median.csv",
        mime="text/csv",
    )

    st.subheader("5. 模拟久期分歧度")
    if duration_disagreement_df.empty:
        st.info("当前无久期分歧度数据。")
    else:
        fig_disagreement = px.line(
            duration_disagreement_df,
            x="date",
            y=["asset_duration_disagreement", "nav_duration_disagreement"],
            markers=True,
            title="模拟久期分歧度（日度久期标准差 / 日度久期均值）",
            labels={
                "value": "分歧度",
                "date": "日期",
                "variable": "久期类型",
            },
        )
        fig_disagreement.update_layout(legend_title_text="久期类型")
        st.plotly_chart(fig_disagreement, use_container_width=True)
        disagreement_csv = duration_disagreement_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="导出模拟久期分歧度 CSV",
            data=disagreement_csv,
            file_name="duration_disagreement.csv",
            mime="text/csv",
        )

    st.subheader("6. 数据表")
    st.caption(f"{selected_fund} 最新 20 条日度杠杆率与久期结果")
    st.dataframe(filtered_duration.sort_values("date").tail(20), use_container_width=True)


def main():
    st.set_page_config(page_title="Bond Fund EKF Duration Estimator", layout="wide")
    st.title("基于 EKF 的债券基金久期动态估计系统")
    st.caption("分步骤交互：数据上传 → 参数设置 → 执行分析 → 结果呈现；支持多基金批量执行。")

    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = None

    st.sidebar.header("步骤1：数据上传")
    nav_file = st.sidebar.file_uploader("上传基金净值 CSV", type=["csv"], key="nav_file")
    index_file = st.sidebar.file_uploader("上传债券指数 CSV", type=["csv"], key="index_file")
    report_file = st.sidebar.file_uploader("上传基金披露 CSV", type=["csv"], key="report_file")

    if not (nav_file and index_file and report_file):
        st.info("请先在左侧完成步骤1的数据上传：fund_nav.csv、bond_index.csv、fund_report.csv。")
        return

    nav_df, index_df, report_df = load_uploaded_data(nav_file, index_file, report_file)
    funds = sorted(nav_df["fund_id"].unique().tolist())
    if not funds:
        st.warning("未找到有效基金代码，请检查净值数据。")
        return

    st.sidebar.success(f"已加载 {len(funds)} 只基金，准备进入参数设置。")
    st.sidebar.caption("执行模式：一次性处理全部基金，结果页面按单基金下拉筛选查看。")

    st.subheader("步骤1：已上传数据")
    st.dataframe(
        pd.DataFrame(
            {
                "基金数": [len(funds)],
                "净值记录数": [len(nav_df)],
                "指数记录数": [len(index_df)],
                "披露记录数": [len(report_df)],
            }
        ),
        use_container_width=True,
    )

    (
        leverage_process_noise,
        weight_process_noise,
        leverage_daily_cap,
        weight_daily_cap,
        observation_noise_matrix,
    ) = render_model_config()

    st.subheader("步骤3：执行分析")
    st.info("执行前确认：全部基金会按同一参数同步计算，分析结果在页面中由基金下拉选择查看。")
    run_analysis = st.button("执行全部基金分析", type="primary")

    if run_analysis:
        daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df, duration_disagreement_df = process_all_funds(
            nav_df,
            index_df,
            report_df,
            duration_map=DEFAULT_DURATION_MAP.copy(),
            leverage_process_noise=leverage_process_noise,
            weight_process_noise=weight_process_noise,
            observation_noise_matrix=observation_noise_matrix,
            leverage_daily_cap=leverage_daily_cap,
            weight_daily_cap=weight_daily_cap,
        )
        if not daily_leverage_df.empty:
            st.session_state.analysis_cache = {
                "daily_leverage_df": daily_leverage_df,
                "daily_allocation_df": daily_allocation_df,
                "daily_duration_df": daily_duration_df,
                "duration_median_df": duration_median_df,
                "duration_disagreement_df": duration_disagreement_df,
                "funds": funds,
            }
            save_outputs(
                daily_leverage_df,
                daily_allocation_df,
                daily_duration_df,
                duration_median_df,
                duration_disagreement_df=duration_disagreement_df,
            )
            st.success(f"分析已完成并保存到 {OUTPUT_DIR} 目录。")

    if st.session_state.analysis_cache is not None:
        cache = st.session_state.analysis_cache
        render_results(
            cache["daily_leverage_df"],
            cache["daily_allocation_df"],
            cache["daily_duration_df"],
            cache["duration_median_df"],
            cache["duration_disagreement_df"],
            cache["funds"],
        )
    else:
        st.info("请先执行全部基金分析，然后可在此处按基金下拉选择结果。")


if __name__ == "__main__":
    main()
