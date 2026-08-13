# Bond Fund Duration EKF Estimator

基于约束 Extended Kalman Filter（EKF）的纯利率债基金日度杠杆率、期限配置与久期动态估计系统。

## 功能概览
- 上传基金每日净值 CSV
- 上传债券综合指数 CSV
- 上传基金披露 CSV
- 估计每日杠杆率
- 估计每日七档期限配置权重
- 估计组合久期和净值久期
- 根据所有基金计算全市场久期中位数
- 用 Streamlit 展示曲线和堆积面积图

## 运行方式
1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 启动应用
   ```bash
   streamlit run app.py
   ```

3. 在页面中上传以下三个文件
   - fund_nav.csv
   - bond_index.csv
   - fund_report.csv

## 输入字段说明
### fund_nav.csv
- fund_id
- date
- nav

### bond_index.csv
- date
- 0_1Y
- 1_3Y
- 3_5Y
- 5_7Y
- 7_10Y
- 10_25Y
- 30Y

### fund_report.csv
- fund_id
- date
- leverage
- duration

注意：
- duration 的 0 表示无披露数据，应在处理时转换为 NaN
- 半年度久期只在二季度末和四季度末有效
- 季度杠杆数据作为观测更新

## 输出
程序会生成以下 CSV：
- daily_leverage.csv
- daily_allocation.csv
- daily_duration.csv
- duration_median.csv

这些文件默认保存到当前目录的 output/ 文件夹中。
