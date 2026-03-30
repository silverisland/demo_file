import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

class DecoupledStationAnalyzer:
    """
    分析并筛选 PV 与 GHI 解耦且现网模型优于自有模型的场站。
    """
    def __init__(self, 
                 pv_data_dir: str, 
                 pv_now_dir: str, 
                 my_pred_path: str,
                 output_dir: str = "./analysis_results"):
        self.pv_data_dir = Path(pv_data_dir)
        self.pv_now_dir = Path(pv_now_dir)
        self.my_pred_path = Path(my_pred_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_station_data(self, station_id: str):
        """加载单个站点的真实数据和现网预报数据"""
        # 匹配文件格式: {id}_pv_data.parquet
        pv_data_file = self.pv_data_dir / f"{station_id}_pv_data.parquet"
        pv_now_file = self.pv_now_dir / f"{station_id}_pv_now.parquet"
        
        if not pv_data_file.exists():
            print(f"Warning: Real data file not found for {station_id}: {pv_data_file}")
            return None
        if not pv_now_file.exists():
            print(f"Warning: Now model file not found for {station_id}: {pv_now_file}")
            return None
            
        df_real = pd.read_parquet(pv_data_file)
        df_now = pd.read_parquet(pv_now_file)
        
        # 检查时间戳列
        if 'timestamp' not in df_real.columns or 'timestamp' not in df_now.columns:
            print(f"Error: 'timestamp' column missing in data for {station_id}")
            return None

        # 确保时间戳格式一致
        df_real['timestamp'] = pd.to_datetime(df_real['timestamp'])
        df_now['timestamp'] = pd.to_datetime(df_now['timestamp'])
        
        # 合并真实数据与现网模型数据
        df = pd.merge(df_real, df_now, on='timestamp', how='inner')
        return df

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """计算向量间的 RMSE"""
        try:
            yt = np.array(y_true, dtype=float)
            yp = np.array(y_pred, dtype=float)
            # 过滤掉包含 NaN 的情况
            mask = ~np.isnan(yt) & ~np.isnan(yp)
            if not np.any(mask): return np.nan
            return np.sqrt(np.mean((yt[mask] - yp[mask])**2))
        except Exception:
            return np.nan

    def get_future_actual_pv(self, df):
        """
        为每个时间点构造未来24小时的真实PV向量。
        """
        # 确保按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        pv_series = df['pv_data'].values
        future_pv_list = []
        for i in range(len(df)):
            if i + 24 <= len(df):
                # 记录从 i+1 到 i+24 的值（对应预报的未来24h）
                future_pv_list.append(pv_series[i:i+24])
            else:
                future_pv_list.append(None)
        df['pv_actual_24h'] = future_pv_list
        return df.dropna(subset=['pv_actual_24h']).copy()

    def run_analysis(self, corr_threshold=0.6, my_error_threshold=None, now_error_improvement=0.2):
        """
        执行分析过程。
        :param corr_threshold: PV-GHI 相关性阈值，低于此值视为解耦。
        :param my_error_threshold: 我的模型误差阈值。如果不传，则动态判断。
        :param now_error_improvement: 现网模型相比我的模型提升的比例（例如 0.2 表示误差降低 20% 以上）。
        """
        if not self.my_pred_path.exists():
            print(f"Error: Prediction file not found at {self.my_pred_path}")
            return
            
        print(f"Loading predictions from {self.my_pred_path}...")
        df_my_all = pd.read_parquet(self.my_pred_path)
        df_my_all['timestamp'] = pd.to_datetime(df_my_all['timestamp'])
        
        station_ids = df_my_all['id'].unique()
        monthly_reports = []

        for sid in station_ids:
            print(f"Analyzing station: {sid}...")
            df_station = self.load_station_data(sid)
            if df_station is None: continue
            
            # 对齐我的模型预测值
            df_my_sid = df_my_all[df_my_all['id'] == sid][['timestamp', 'prediction']]
            df = pd.merge(df_station, df_my_sid, on='timestamp', how='inner')
            
            if df.empty:
                print(f"No aligned data for station {sid}")
                continue
            
            # 构造未来24h真值
            df = self.get_future_actual_pv(df)
            
            # 计算预测误差
            df['my_rmse'] = df.apply(lambda row: self.calculate_rmse(row['pv_actual_24h'], row['prediction']), axis=1)
            df['now_rmse'] = df.apply(lambda row: self.calculate_rmse(row['pv_actual_24h'], row['pv_predict_now']), axis=1)
            
            # 按月分析
            df['month'] = df['timestamp'].dt.to_period('M')
            
            for month, group in df.groupby('month'):
                # 核心逻辑：解耦度计算
                # 仅在有辐射的时间段计算相关性，更真实反映响应关系
                daylight = group[group['ghi'] > 0.05 * group['ghi'].max()] 
                if len(daylight) < 24: # 样本量太少则跳过
                    continue
                
                pv_ghi_corr = daylight['pv_data'].corr(daylight['ghi'])
                
                # 计算平均误差
                avg_my_rmse = group['my_rmse'].mean()
                avg_now_rmse = group['now_rmse'].mean()
                
                if np.isnan(pv_ghi_corr) or np.isnan(avg_my_rmse) or np.isnan(avg_now_rmse):
                    continue

                # 判定条件
                is_decoupled = pv_ghi_corr < corr_threshold
                
                # 现网模型提升判定：现网误差显著更小
                better_than_my = avg_now_rmse < avg_my_rmse * (1 - now_error_improvement)
                
                # 如果指定了硬阈值
                is_my_bad = True
                if my_error_threshold is not None:
                    is_my_bad = avg_my_rmse > my_error_threshold

                should_switch = is_decoupled and better_than_my and is_my_bad
                
                monthly_reports.append({
                    'id': sid,
                    'month': str(month),
                    'pv_ghi_corr': round(pv_ghi_corr, 4),
                    'my_rmse': round(avg_my_rmse, 4),
                    'now_rmse': round(avg_now_rmse, 4),
                    'is_decoupled': is_decoupled,
                    'better_performance': "NowModel" if better_than_my else "MyModel",
                    'recommendation': "SwitchToNow" if should_switch else "KeepMyModel"
                })

        if not monthly_reports:
            print("No analysis results generated. Check data overlap and paths.")
            return

        report_df = pd.DataFrame(monthly_reports)
        
        # 1. 保存完整报告
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d')}.csv"
        report_df.to_csv(report_path, index=False)
        
        # 2. 导出下个月建议切换的清单
        # 获取最近一个月份的数据作为参考
        latest_month = sorted(report_df['month'].unique())[-1]
        switch_list = report_df[(report_df['month'] == latest_month) & 
                                (report_df['recommendation'] == "SwitchToNow")]
        
        switch_list_path = self.output_dir / f"suggested_switch_list_{latest_month}.csv"
        switch_list.to_csv(switch_list_path, index=False)
        
        print("-" * 30)
        print(f"Analysis Summary for {latest_month}:")
        print(f"Total stations analyzed: {len(report_df[report_df['month'] == latest_month])}")
        print(f"Stations recommended to switch: {len(switch_list)}")
        print(f"Full report: {report_path}")
        print(f"Switch list: {switch_list_path}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 请根据你的实际目录结构修改以下路径
    PV_DATA_FOLDER = "./data/pv_data_dir"    # 存放 {id}_pv_data.parquet 的文件夹
    PV_NOW_FOLDER = "./data/pv_now_dir"      # 存放 {id}_pv_now.parquet 的文件夹
    MY_PREDICTION_FILE = "./mock_power_data.parquet" # 我的模型预测汇总文件
    
    # 实例化并运行
    analyzer = DecoupledStationAnalyzer(
        pv_data_dir=PV_DATA_FOLDER,
        pv_now_dir=PV_NOW_FOLDER,
        my_pred_path=MY_PREDICTION_FILE
    )
    
    analyzer.run_analysis(
        corr_threshold=0.5,        # PV-GHI相关性低于0.5认为解耦
        now_error_improvement=0.15 # 现网模型误差比我低15%以上时考虑切换
    )
