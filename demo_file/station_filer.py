import pandas as pd
import numpy as np
from typing import List, Tuple, Literal

class PhaseShiftDetector:
    def __init__(self, 
                 target_col: str,
                 reference_col: str = None, 
                 time_col: str = 'timestamp',
                 id_col: str = 'id',
                 max_shift_hours: int = 12,
                 shift_threshold: int = 3):
        """
        相位偏差检测器。用于检测存在明显时序平移（例如差了12小时）的异常站点。
        
        :param target_col: 待检测的列名（例如 'GHI' 或 'GHI_future1d'）
        :param reference_col: 参照物列名（例如正确的 'pv'），若适用绝对峰值法可不传。
        :param time_col: 时间戳列名。
        :param id_col: 场站ID列名。
        :param max_shift_hours: 允许搜索的最大平移小时数，默认12。
        :param shift_threshold: 偏差大于等于多少小时被认为是有明显相位偏差。
        """
        self.target_col = target_col
        self.reference_col = reference_col
        self.time_col = time_col
        self.id_col = id_col
        self.max_shift_hours = max_shift_hours
        self.shift_threshold = shift_threshold

    def _calculate_lag_cross_correlation(self, ref_arr: np.ndarray, target_arr: np.ndarray) -> int:
        """
        使用滑窗互相关求滞后值。
        计算 target_arr 相对于 ref_arr 滞后了多少个小时。
        如果 target 每天曲线峰值发生在晚上，而 ref 发生在正午，这能准确计算出平移量。
        """
        ref_std = np.std(ref_arr)
        target_std = np.std(target_arr)
        if ref_std == 0 or target_std == 0:
            return 0
            
        best_lag = 0
        max_corr = -np.inf
        
        # 在给定的允许范围内进行左右平移寻找最大皮尔逊相关系数
        for lag in range(-self.max_shift_hours, self.max_shift_hours + 1):
            if lag < 0:
                # target 曲线超前
                corr = np.corrcoef(ref_arr[:lag], target_arr[-lag:])[0, 1]
            elif lag > 0:
                # target 曲线滞后
                corr = np.corrcoef(ref_arr[lag:], target_arr[:-lag])[0, 1]
            else:
                corr = np.corrcoef(ref_arr, target_arr)[0, 1]
                
            if np.isnan(corr):
                continue
                
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
                
        return best_lag

    def _detect_absolute_phase_by_peak(self, arr: np.ndarray, expected_peak_hour: int = 12) -> int:
        """
        绝对峰值法：按天折叠并计算日内平均曲线，寻找峰值的绝对发生时间以检测错位。
        无论对 GHI 还是 PV 都非常有效，因为它们总是应该在系统时间的正午前后达到峰顶。
        """
        if len(arr) % 24 != 0:
            raise ValueError("数组长度必须是 24 的倍数以便按天评估")
            
        days = len(arr) // 24
        reshaped = arr.reshape(days, 24)
        daily_mean = np.mean(reshaped, axis=0)  # shape: (24,)
        
        # 找到这 24 小时平均趋势中峰值所在的最大值索引
        peak_hour = int(np.argmax(daily_mean))
        
        # 计算与预期的偏差
        diff = peak_hour - expected_peak_hour
        
        # 转换至 -12 到 +12 的循环环形空间中
        if diff > 12:
            diff -= 24
        elif diff < -12:
            diff += 24
            
        return diff

    def detect_bad_stations(self, df: pd.DataFrame, method: Literal['cross_corr', 'peak_hour'] = 'cross_corr') -> List[str]:
        """
        遍历各站点并根据指定的探测法过滤出有严重相位误差的站点。
        """
        abnormal_stations = set()
        
        for station_id, group in df.groupby(self.id_col):
            shifts = []
            
            # 使用时间戳排序保证序列正确
            if self.time_col in group.columns:
                group = group.sort_values(self.time_col)
                
            if method == 'cross_corr':
                if not self.reference_col:
                    raise ValueError("使用交叉互相关法 (cross_corr) 需要指定 reference_col")
                    
                target_data = group[self.target_col].values
                ref_data = group[self.reference_col].values
                
                valid_mask = ~pd.isna(target_data) & ~pd.isna(ref_data)
                target_data_clean = target_data[valid_mask].astype(float)
                ref_data_clean = ref_data[valid_mask].astype(float)
                
                # 至少要有 24 个点以上才能稳定测出平移
                if len(target_data_clean) < 24:
                    continue
                    
                # 按照24的一个整型小段进行切片计算多个 shift，以维持中位数鲁棒策略
                chunks = len(target_data_clean) // 24
                for i in range(chunks):
                    t_chunk = target_data_clean[i*24:(i+1)*24]
                    r_chunk = ref_data_clean[i*24:(i+1)*24]
                    if np.std(t_chunk) == 0 or np.std(r_chunk) == 0:
                        continue
                    shift = self._calculate_lag_cross_correlation(r_chunk, t_chunk)
                    shifts.append(shift)
                    
            elif method == 'peak_hour':
                # 判断当前 target_col 中存放的是单一浮点数，还是数组（诸如 GHI_future1d）
                is_array_col = False
                
                # 情况A: GHI_future1d (每行一个是长度24的array预报)
                for _, row in group.iterrows():
                    val = row[self.target_col]
                    if isinstance(val, (np.ndarray, list)):
                        is_array_col = True
                        target_data = np.array(val, dtype=float)
                        if len(target_data) == 24 and not np.isnan(target_data).any():
                            # 对于这种本来就是 24 小时一天的新预测数据，一行直接做校验
                            shift = self._detect_absolute_phase_by_peak(target_data, expected_peak_hour=12)
                            shifts.append(shift)
                            
                # 情况B: GHI (在时序里是一列浮点序列)
                if not is_array_col:
                    target_data = group[self.target_col].values
                    target_data_clean = target_data[~pd.isna(target_data)].astype(float)
                    
                    chunks = len(target_data_clean) // 24
                    for i in range(chunks):
                        t_chunk = target_data_clean[i*24:(i+1)*24]
                        if np.std(t_chunk) == 0:
                            continue
                        shift = self._detect_absolute_phase_by_peak(t_chunk, expected_peak_hour=12)
                        shifts.append(shift)
            
            if not shifts:
                continue
                
            # 使用中位数代表这个站点的整体偏差特征
            median_shift = np.median(shifts)
            
            if abs(median_shift) >= self.shift_threshold:
                abnormal_stations.add(station_id)
                
        return list(abnormal_stations)


def filter_stations_with_phase_shift(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    业务逻辑主出口：清洗含有相位误差的站点。
    返回剔除错误站点后的 DataFrame 和被筛掉的站点 ID 列表。
    """
    # 策略 1：检查历史 GHI 相对于 pv 是否存在严重的滞后或超前
    # 适配新版单点数值格式，通过 time_col='timestamp' 对齐
    history_detector = PhaseShiftDetector(
        target_col='GHI',
        reference_col='pv',
        time_col='timestamp',
        id_col='id',
        shift_threshold=4  # 超过 4 小时偏差剔除
    )
    bad_stations_hist = history_detector.detect_bad_stations(df, method='cross_corr')
    
    # 策略 2：检查未来的气象预报本身(长度24的array)是否存在非常离谱的时序问题
    # 不需要 reference_col，直接利用峰值对齐
    future_detector = PhaseShiftDetector(
        target_col='GHI_future1d',
        time_col='timestamp',
        id_col='id',
        shift_threshold=4
    )
    bad_stations_future = future_detector.detect_bad_stations(df, method='peak_hour')
    
    # 任何一种诊断出相位问题统统淘汰
    all_bad_stations = list(set(bad_stations_hist).union(set(bad_stations_future)))
    
    # 清理掉这些站点
    df_clean = df[~df['id'].isin(all_bad_stations)].copy()
    
    return df_clean, all_bad_stations

if __name__ == '__main__':
    # 简单的本地测试环节
    np.random.seed(42)
    stations = [1, 2]
    data = []
    
    timestamps = pd.date_range(start="2024-06-01", periods=168, freq='h')
    
    for st_id in stations:
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # 钟形基础辐射曲线
            base_curve = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) if 6 <= hour <= 18 else 0.0
            pv_current = base_curve * 50
            
            if st_id == 2:
                # 站点 2：我们人工制造时效平移故障（延迟6小时）
                shifted_hour = (hour - 6) % 24
                ghi_curve = np.clip(np.sin(np.pi * (shifted_hour - 6) / 12), 0, None) if 6 <= shifted_hour <= 18 else 0.0
                ghi_current = ghi_curve * 1000
                future_array = np.array([ghi_current]*24) # 只是给个凑数的假数组
            else:
                # 站点 1：正常情况
                ghi_current = base_curve * 1000
                
                # 获取该时刻往后24小时的预报模拟
                future_array = []
                for fh in range(24):
                    fh_hour = (hour + fh) % 24
                    f_curve = np.clip(np.sin(np.pi * (fh_hour - 6) / 12), 0, None) if 6 <= fh_hour <= 18 else 0.0
                    future_array.append(f_curve * 1000)
                future_array = np.array(future_array)
            
            data.append({
                'id': st_id,
                'timestamp': ts,
                'pv': pv_current,
                'GHI': ghi_current,
                'GHI_future1d': future_array
            })
            
    test_df = pd.DataFrame(data)
    
    print("清洗前包含站点：", test_df['id'].unique())
    df_clean, bad_stations = filter_stations_with_phase_shift(test_df)
    
    print("由于相位漂移严重而被剔除的坏站列表:", bad_stations)
    print("清洗后保留可用站点:", df_clean['id'].unique())
