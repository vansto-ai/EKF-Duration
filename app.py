import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DEFAULT_DURATION_MAP, ALLOCATION_COLUMNS
from src.data_utils import load_uploaded_data, prepare_fund_data
from src.ekf_model import estimate_daily_fund_states


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_outputs(daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df):
    daily_leverage_df.to_csv(OUTPUT_DIR / "daily_leverage.csv", index=False)
    daily_allocation_df.to_csv(OUTPUT_DIR / "daily_allocation.csv", index=False)
    daily_duration_df.to_csv(OUTPUT_DIR / "daily_duration.csv", index=False)
    duration_median_df.to_csv(OUTPUT_DIR / "duration_median.csv", index=False)


def process_all_funds(nav_df, index_df, report_df):
    funds = sorted(nav_df["fund_id"].unique().tolist())
    all_daily = []

    for fund_id in funds:
        fund_df = prepare_fund_data(fund_id, nav_df, index_df, report_df)
        fund_result = estimate_daily_fund_states(fund_df, DEFAULT_DURATION_MAP, report_df)
        all_daily.append(fund_result)

    if not all_daily:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_daily_df = pd.concat(all_daily, ignore_index=True)
    all_daily_df = all_daily_df.sort_values(["fund_id", "date"]).reset_index(drop=True)

    # 日度杠杆率
    daily_leverage_df = all_daily_df[["fund_id", "date", "leverage"]].copy()
    daily_leverage_df = daily_leverage_df.rename(columns={"leverage": "leverage"})

    # 日度期限配置
    daily_allocation_df = all_daily_df[["fund_id", "date"] + [c for c in all_daily_df.columns if c.endswith("_weight")]].copy()
    daily_allocation_df = daily_allocation_df.rename(columns={
        "0_1Y_weight": "0_1Y_weight",
        "1_3Y_weight": "1_3Y_weight",
        "3_5Y_weight": "3_5Y_weight",
        "5_7Y_weight": "5_7Y_weight",
        "7_10Y_weight": "7_10Y_weight",
        "10_25Y_weight": "10_25Y_weight",
        "30Y_weight": "30Y_weight",
    })

    # 日度久期
    daily_duration_df = all_daily_df[["fund_id", "date", "asset_duration", "nav_duration", "leverage"]].copy()
    daily_duration_df = daily_duration_df.rename(columns={
        "asset_duration": "asset_duration",
        "nav_duration": "nav_duration",
        "leverage": "leverage",
    })

    # 市场久期中位数
    duration_median_df = (
        all_daily_df.groupby("date", as_index=False)[["asset_duration", "nav_duration"]]
        .median()
        .rename(columns={
            "asset_duration": "asset_duration_median",
            "nav_duration": "nav_duration_median",
        })
    )

    return daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df


def main():
    st.set_page_config(page_title="Bond Fund EKF Duration Estimator", layout="wide")

    st.title("基于 EKF 的债券基金久期动态估计系统")

    st.sidebar.header("数据上传")
    nav_file = st.sidebar.file_uploader("上传基金净值 CSV", type=["csv"])
    index_file = st.sidebar.file_uploader("上传债券指数 CSV", type=["csv"])
    report_file = st.sidebar.file_uploader("上传基金披露 CSV", type=["csv"])

    if nav_file and index_file and report_file:
        nav_df, index_df, report_df = load_uploaded_data(nav_file, index_file, report_file)
        funds = sorted(nav_df["fund_id"].unique().tolist())

        if not funds:
            st.warning("未找到有效基金代码，请检查净值数据。")
            return

        selected_fund = st.selectbox("选择基金", funds)

        # 处理全部基金并得到全市场中位数
        daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df = process_all_funds(
            nav_df, index_df, report_df
        )

        # 选择单基金结果
        selected_daily = daily_duration_df[daily_duration_df["fund_id"] == selected_fund].sort_values("date").reset_index(drop=True)
        selected_leverage = daily_leverage_df[daily_leverage_df["fund_id"] == selected_fund].sort_values("date").reset_index(drop=True)
        selected_alloc = daily_allocation_df[daily_allocation_df["fund_id"] == selected_fund].sort_values("date").reset_index(drop=True)

        st.success("数据已加载并成功完成 EKF 滤波估计。")

        # 生成输出文件
        save_outputs(daily_leverage_df, daily_allocation_df, daily_duration_df, duration_median_df)
        st.caption(f"已保存输出文件至 {OUTPUT_DIR.resolve()}")

        # 展示选择基金的关键指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均杠杆率", round(selected_leverage["leverage"].mean(), 4))
        with col2:
            st.metric("平均组合久期", round(selected_daily["asset_duration"].mean(), 2))
        with col3:
            st.metric("平均净值久期", round(selected_daily["nav_duration"].mean(), 2))

        # 图1: 杠杆率曲线
        st.subheader("图1：杠杆率曲线")
        fig_leverage = px.line(
            selected_leverage,
            x="date",
            y="leverage",
            markers=True,
            title=f"{selected_fund} 日度杠杆率",
        )
        fig_leverage.add_hline(1.0, line_dash="dash", line_color="gray")
        fig_leverage.add_hline(1.4, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_leverage, use_container_width=True)

        # 图2: 组合久期和净值久期
        st.subheader("图2：组合久期和净值久期")
        fig_duration = px.line(
            selected_daily,
            x="date",
            y=["asset_duration", "nav_duration"],
            markers=True,
            title=f"{selected_fund} 资产久期 / 净值久期",
        )
        st.plotly_chart(fig_duration, use_container_width=True)

        # 图3: 期限配置变化堆积面积图
        st.subheader("图3：期限配置变化")
        area_data = selected_alloc.copy()
        area_data = area_data.melt(id_vars=["fund_id", "date"], var_name="bucket", value_name="weight")
        fig_area = px.area(
            area_data,
            x="date",
            y="weight",
            color="bucket",
            title=f"{selected_fund} 七档期限配置权重",
            category_orders={"bucket": ["0_1Y_weight", "1_3Y_weight", "3_5Y_weight", "5_7Y_weight", "7_10Y_weight", "10_25Y_weight", "30Y_weight"]},
        )
        st.plotly_chart(fig_area, use_container_width=True)

        # 图4: 全市场久期中位数
        st.subheader("图4：全市场久期中位数")
        median_fig = px.line(
            duration_median_df,
            x="date",
            y=["asset_duration_median", "nav_duration_median"],
            markers=True,
            title="全市场组合久期与净值久期中位数",
        )
        st.plotly_chart(median_fig, use_container_width=True)

        # 数据表展示
        st.subheader("日度杠杆率")
        st.dataframe(selected_leverage.tail(20), use_container_width=True)

        st.subheader("日度久期")
        st.dataframe(selected_daily.tail(20), use_container_width=True)

    else:
        st.info("请先在左侧上传 fund_nav.csv、bond_index.csv 与 fund_report.csv。")


if __name__ == "__main__":
    main()
