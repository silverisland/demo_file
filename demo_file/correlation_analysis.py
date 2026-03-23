import numpy as np
import pandas as pd
from typing import List, Tuple

class StationCorrelationAnalyzer:
    """
    用于分析并筛选光伏(PV)与气象(GHI)数据相关性极低的场站。
    
    背景：
    如果直接对长序列计算皮尔逊相关系数，由于白天和黑夜的必然交替，
    不论阴天晴天，相关性都会被日夜的绝对0值规律强行拉高（出现虚假的高相关）。
    
    核心逻辑（基于新数据格式）：
    直接根据 timestamp 按天进行聚合，提取每天的“日累计发电量”与“日总辐射量”（或日峰值），
    并在“每日”的维度上计算这二者的真实相关系数。
    由于 GHI/PV 已经变成了对应时刻的单浮点数值，此前的 reshape 逻辑不再需要。
    """
    
    def __init__(self, 
                 pv_col: str = 'pv', 
                 ghi_col: str = 'GHI',
                 station_id_col: str = 'id',
                 time_col: str = 'timestamp',
                 metric: str = 'sum'):
        """
        初始化相关性筛选器
        
        :param pv_col: 光伏历史数据列名，为单个浮点值
        :param ghi_col: GHI历史数据列名，为单个浮点值
        :param station_id_col: 场站ID列名
        :param time_col: 时间戳列名
        :param metric: 日维度的提取特征方式，支持 'sum' (日累积量) 和 'max' (日峰值)
        """
        self.pv_col = pv_col
        self.ghi_col = ghi_col
        self.station_id_col = station_id_col
        self.time_col = time_col
        self.metric = metric
        
    def evaluate_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        评估每个场站的 PV 与 GHI 真实相关性。
        :return: 包含各个场站相关性得分的 DataFrame
        """
        # 提取相关列，避免修改原数据（这里也默认传入的 dataframe 中包含了这些列）
        df_calc = df[[self.station_id_col, self.time_col, self.pv_col, self.ghi_col]].copy()
        
        # 确保时间列为 datetime 格式
        df_calc[self.time_col] = pd.to_datetime(df_calc[self.time_col])
        
        # 提取出具体的日期（年月日），用于每日的分组聚合计算
        df_calc['date'] = df_calc[self.time_col].dt.date
        
        # 按场站和日期进行分组，计算每日的 sum 或 max 聚合值
        daily_df = df_calc.groupby([self.station_id_col, 'date'])[[self.pv_col, self.ghi_col]].agg(self.metric).reset_index()
        
        results = []
        station_groups = daily_df.groupby(self.station_id_col)
        
        for station_id, group_df in station_groups:
            all_daily_pv = group_df[self.pv_col].values
            all_daily_ghi = group_df[self.ghi_col].values
            
            # 计算皮尔逊相关系数
            if len(all_daily_pv) > 1:
                # np.corrcoef 返回一个 2x2 的协方差矩阵，取右上角 [0, 1] 即为两组变量的相关系数
                corr = np.corrcoef(all_daily_pv, all_daily_ghi)[0, 1]
            else:
                corr = np.nan
            
            # 统计当前站点包含多少条原始时序样本
            sample_count = len(df_calc[df_calc[self.station_id_col] == station_id])
                
            results.append({
                'station_id': station_id,
                'sample_count': sample_count,
                'days_count': len(all_daily_pv),
                # 为了显示好看点，保留四位小数
                'pv_ghi_corr': round(corr, 4) if not np.isnan(corr) else np.nan
            })
            
        result_df = pd.DataFrame(results).sort_values(by='pv_ghi_corr', ascending=False)
        return result_df

    def filter_stations(self, df: pd.DataFrame, threshold: float = 0.6) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        """
        过滤出低相关性站点，以便移交给单变量预测模型。
        
        :param threshold: 皮尔逊相关性低于此阈值的站点将被剔除
        :return: (高相关性站点的df, 低相关性站点的df, 低相关性站点ID列表)
        """
        corr_df = self.evaluate_stations(df)
        
        # 找出坏站
        bad_stations = corr_df[corr_df['pv_ghi_corr'] < threshold]['station_id'].tolist()
        
        # 基于坏站列表对原 df 进行拆分
        good_df = df[~df[self.station_id_col].isin(bad_stations)].copy()
        bad_df = df[df[self.station_id_col].isin(bad_stations)].copy()
        
        return good_df, bad_df, bad_stations

if __name__ == '__main__':
    # ---------------- 模拟测试环节（已适配按单一样本行的时间序列格式） ----------------
    np.random.seed(42)
    
    stations = [1, 2, 3] # 三个场站
    
    data = []
    
    for st_id in stations:
        # 给每个场站分别生成连续 7 天的小时级别时间戳序列 (总共 168 个时刻)
        timestamps = pd.date_range(start="2024-06-01", periods=168, freq='h')
        daily_weather_factors = np.random.uniform(0.2, 1.0, 7)
        
        for day_idx in range(7):
            day_factor = daily_weather_factors[day_idx]
            
            # 捏造一天内光照周期的钟形曲线 (0~23点)
            base_curve = np.sin(np.linspace(0, np.pi, 24))
            base_curve = np.clip(base_curve, 0, None)
            
            for h in range(24):
                ts = timestamps[day_idx * 24 + h]
                
                # 当前时刻 GHI 单浮点数
                ghi_current = base_curve[h] * 1000 * day_factor
                
                # 当前时刻的预测特征 GHI_future1d: 一个包含未来24小时预报的 np.array
                # (相关系数计算中其实不依赖它，但这里演示作为 Dataframe 的一列)
                future_array = np.clip(np.random.normal(ghi_current, 50, 24), 0, None)
                
                if st_id == 1:
                    # 站点 1：好站，PV 随着 GHI 的天气好坏强相关同增同减
                    pv_current = base_curve[h] * 50 * day_factor * np.random.uniform(0.9, 1.1)
                elif st_id == 2:
                    # 站点 2：坏站，发不发电全看心情，GHI是晴是雨对它毫无影响（完全脱敏）
                    pv_current = base_curve[h] * 50 * np.random.uniform(0.2, 1.0)
                else:
                    # 站点 3：坏站，表现平庸且部分时间限电，相关性也随机且偏低
                    pv_current = base_curve[h] * 50 * 0.4 * np.random.uniform(0.9, 1.1)
                    
                data.append({
                    'id': st_id,
                    'timestamp': ts,
                    'pv': pv_current,
                    'GHI': ghi_current,
                    'GHI_future1d': future_array
                })
            
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("已生成最新的行级时序DataFrame。数据前两行概览：")
    # 打印时不打印 GHI_future1d，因为太长了影响阅读
    view_df = df.drop(columns=['GHI_future1d'])
    print(view_df.head(2).to_string())
    
    print("\n正在计算各大场站 PV 与 GHI 的日级别真实相关性...")
    # 初始化时注意由于我们的单点 PV 列名变为 'pv'，此处明确指出
    analyzer = StationCorrelationAnalyzer(pv_col='pv', ghi_col='GHI', time_col='timestamp', metric='sum')
    corr_df = analyzer.evaluate_stations(df)
    
    print("\n相关性评分总览：")
    print(corr_df.to_string(index=False))
    
    # 低于 0.7 的视为坏站，交给其余单变量模型
    good_df, bad_df, bad_stations = analyzer.filter_stations(df, threshold=0.7)
    
    print("\n" + "=" * 60)
    print(f"设定相关性阈值: 0.7")
    print(f"被剔除留作后续【单变量预测】的异常站点ID: {bad_stations}")
    print(f"剩余留在当前预测池内的优秀样本(行)数: {len(good_df)}")
    print(f"已被分离剥离出来的低相关样本(行)数: {len(bad_df)}")
