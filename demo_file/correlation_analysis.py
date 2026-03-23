import numpy as np
import pandas as pd
from typing import List, Tuple

class StationCorrelationAnalyzer:
    """
    用于分析并筛选光伏(PV)与气象(GHI)数据相关性极低的场站。
    
    背景：
    如果直接对168小时的长序列计算皮尔逊相关系数，由于白天和黑夜的必然交替，
    不论阴天晴天，相关性都会被日夜的绝对0值规律强行拉高（出现虚假的高相关）。
    
    核心逻辑：
    为了真正捕捉“GHI是晴天，但PV表现却像阴天”这种幅度变化不一致的站，
    我们必须完全屏蔽日夜周期，直接提取每天的“日累计发电量”与“日总辐射量”（或日峰值），
    并在“每日”的维度上计算这二者的真实相关系数。
    """
    
    def __init__(self, 
                 pv_col: str = 'pv_data_history', 
                 ghi_col: str = 'GHI_history',
                 station_id_col: str = 'id',
                 metric: str = 'sum'):
        """
        初始化相关性筛选器
        
        :param pv_col: 光伏历史数据列名，假设每行是一个长度为 7*24 的 np.array
        :param ghi_col: GHI历史数据列名，同样也是 7*24
        :param station_id_col: 场站ID列名
        :param metric: 日维度的提取特征方式，支持 'sum' (日累积量) 和 'max' (日峰值)
        """
        self.pv_col = pv_col
        self.ghi_col = ghi_col
        self.station_id_col = station_id_col
        self.metric = metric
        
    def _extract_daily_metrics(self, arr: np.ndarray) -> np.ndarray:
        """从 168 小时的序列中提取 7 天的每日特征"""
        if len(arr) % 24 != 0:
            raise ValueError(f"数组长度 {len(arr)} 不是24的整数倍")
            
        reshaped = arr.reshape(-1, 24)
        if self.metric == 'sum':
            return np.sum(reshaped, axis=1)
        elif self.metric == 'max':
            return np.max(reshaped, axis=1)
        else:
            raise ValueError(f"不支持的 metric: {self.metric}")

    def evaluate_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        评估每个场站的 PV 与 GHI 真实相关性。
        :return: 包含各个场站相关性得分的 DataFrame
        """
        station_groups = df.groupby(self.station_id_col)
        
        results = []
        
        for station_id, group_df in station_groups:
            all_daily_pv = []
            all_daily_ghi = []
            
            for pv_arr, ghi_arr in zip(group_df[self.pv_col], group_df[self.ghi_col]):
                all_daily_pv.extend(self._extract_daily_metrics(pv_arr))
                all_daily_ghi.extend(self._extract_daily_metrics(ghi_arr))
                
            # 计算皮尔逊相关系数
            if len(all_daily_pv) > 1:
                # np.corrcoef 返回一个 2x2 的协方差矩阵，取右上角 [0, 1] 即为两组变量的相关系数
                corr = np.corrcoef(all_daily_pv, all_daily_ghi)[0, 1]
            else:
                corr = np.nan
                
            results.append({
                'station_id': station_id,
                'sample_count': len(group_df),
                'days_count': len(all_daily_pv),
                ... # 为了显示好看点，保留四位小数
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
        
        # 拆分数据
        good_df = df[~df[self.station_id_col].isin(bad_stations)].copy()
        bad_df = df[df[self.station_id_col].isin(bad_stations)].copy()
        
        return good_df, bad_df, bad_stations

if __name__ == '__main__':
    # ---------------- 模拟测试环节 ----------------
    np.random.seed(42)
    
    n_samples_per_station = 10
    stations = [1, 2, 3] # 三个场站
    
    data = []
    
    for st_id in stations:
        for _ in range(n_samples_per_station):
            # 随机模拟这 7 天的天气好坏，系数范围 0.2(暴雨) ~ 1.0(极晴)
            daily_weather_factors = np.random.uniform(0.2, 1.0, 7)
            
            ghi_168 = []
            pv_168 = []
            
            for day_factor in daily_weather_factors:
                # 捏造一天内光照周期的钟形曲线 (0~23点)
                base_curve = np.sin(np.linspace(0, np.pi, 24))
                base_curve = np.clip(base_curve, 0, None)
                
                # GHI 总是老老实实跟着天气预报走
                ghi_day = base_curve * 1000 * day_factor
                
                if st_id == 1:
                    # 站点 1：好站，PV 随着 GHI 的天气好坏强相关同增同减
                    pv_day = base_curve * 50 * day_factor * np.random.uniform(0.9, 1.1)
                elif st_id == 2:
                    # 站点 2：坏站，发不发电全看心情，GHI是晴是雨对它毫无影响（完全脱敏）
                    pv_day = base_curve * 50 * np.random.uniform(0.2, 1.0)
                else:
                    # 站点 3：坏站，部分时间限电或异常，表现极为均质躺平，相关性也极低
                    pv_day = base_curve * 50 * 0.4 * np.random.uniform(0.9, 1.1)
                    
                ghi_168.extend(ghi_day)
                pv_168.extend(pv_day)
                
            data.append({
                'id': st_id,
                'pv_data_history': np.array(pv_168),
                'GHI_history': np.array(ghi_168)
            })
            
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("正在计算各大场站 PV 与 GHI 的日级别真实相关性...")
    analyzer = StationCorrelationAnalyzer(metric='sum')
    corr_df = analyzer.evaluate_stations(df)
    
    print("\n相关性评分总览：")
    print(corr_df.to_string(index=False))
    
    # 低于 0.7 的视为坏站，交给其余单变量模型
    good_df, bad_df, bad_stations = analyzer.filter_stations(df, threshold=0.7)
    
    print("\n" + "=" * 60)
    print(f"设定相关性阈值: 0.7")
    print(f"被剔除留作后续【单变量预测】的异常站点ID: {bad_stations}")
    print(f"剩余留在当前多波段深度学习预测池内的优秀样本数: {len(good_df)}")
    print(f"已被分离剥离出来的低相关样本数: {len(bad_df)}")
