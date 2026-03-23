import pandas as pd
import numpy as np
from typing import List, Tuple, Literal

class PhaseShiftDetector:
    def __init__(self, 
                 target_col: str,
                 reference_col: str = None, 
                 max_shift_hours: int = 12,
                 shift_threshold: int = 3):
        """
        相位偏差检测器。用于检测存在明显时序平移（例如差了12小时）的异常站点。
        
        :param target_col: 待检测的列名（例如 'GHI_history' 或 'GHI_future1d'）
        :param reference_col: 参照物列名（例如正确的 'pv_data_history'），若适用绝对峰值法可不传。
        :param max_shift_hours: 允许搜索的最大平移小时数，考虑到半天是12小时，默认12。
        :param shift_threshold: 偏差大于等于多少小时被认为是不合格站点（有明显相位偏差）。
        """
        self.target_col = target_col
        self.reference_col = reference_col
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
        无论对 GHI 还是 PV 都非常有效，因为它们理论上总是应该在系统时间的正午前后达到峰顶。
        """
        if len(arr) % 24 != 0:
            raise ValueError("数组长度必须是 24 的倍数以便按天折叠")
            
        days = len(arr) // 24
        reshaped = arr.reshape(days, 24)
        daily_mean = np.mean(reshaped, axis=0)  # shape: (24,)
        
        # 找到这 24 小时平均趋势中峰值所在的最大值
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
        
        :param df: 输入的 dataframe
        :param method: 
            - 'cross_corr': 互相关法，需要 reference_col。用来检查 GHI 和同行的 PV 的相位错位。
            - 'peak_hour' : 绝对峰值法。看 GHI 单条序列自己折叠平均后，最高点是不是偏离正午。
        """
        abnormal_stations = set()
        
        for station_id, group in df.groupby('id'):
            shifts = []
            
            for _, row in group.iterrows():
                target_data = np.array(row[self.target_col], dtype=float)
                
                # 若包含 nan 则忽略此次对比
                if np.isnan(target_data).any():
                    continue
                
                if method == 'cross_corr':
                    if not self.reference_col:
                        raise ValueError("使用交叉互相关法 (cross_corr) 需要指定 reference_col")
                    ref_data = np.array(row[self.reference_col], dtype=float)
                    if len(ref_data) != len(target_data) or np.isnan(ref_data).any():
                        continue
                        
                    shift = self._calculate_lag_cross_correlation(ref_data, target_data)
                    shifts.append(shift)
                    
                elif method == 'peak_hour':
                    shift = self._detect_absolute_phase_by_peak(target_data, expected_peak_hour=12)
                    shifts.append(shift)
            
            if not shifts:
                continue
                
            # 使用中位数代表这个站点的整体偏差特征，能够忽略掉个别阴雨天等毛刺所带来的误导
            median_shift = np.median(shifts)
            
            if abs(median_shift) >= self.shift_threshold:
                abnormal_stations.add(station_id)
                
        return list(abnormal_stations)


def filter_stations_with_phase_shift(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    业务逻辑主出口：清洗含有相位误差的站点，例如 GHI 和 PV 数据产生 12 小时级位移的废弃数据。
    返回剔除错误站点后的 DataFrame 和被筛掉的站点 ID 列表。
    """
    
    # 策略 1：检查 GHI_history 相对于 pv_data_history 是否存在严重的滞后或超前
    #（因为 PV 功率跟气象 GHI 必须也是强同相的）
    history_detector = PhaseShiftDetector(
        target_col='GHI_history',
        reference_col='pv_data_history', 
        shift_threshold=4  # 阈值可按需调整，例如超过 4 小时偏差直接干掉
    )
    bad_stations_hist = history_detector.detect_bad_stations(df, method='cross_corr')
    
    # 策略 2：检查未来的气象 GHI_future1d 自己本身是否存在日落巅峰/半夜出太阳的错乱相位
    future_detector = PhaseShiftDetector(
        target_col='GHI_future1d',
        shift_threshold=4
    )
    # 因为未来也许没有 pv 做参照，这里使用单独计算自己正午绝对峰值的校验法
    bad_stations_future = future_detector.detect_bad_stations(df, method='peak_hour')
    
    # 取并集：只要在任何一段发现有时区/相位问题的站点统统淘汰
    all_bad_stations = list(set(bad_stations_hist).union(set(bad_stations_future)))
    
    # 清理掉这些站点
    df_clean = df[~df['id'].isin(all_bad_stations)].copy()
    
    return df_clean, all_bad_stations
