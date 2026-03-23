import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

class TimeSeriesAnomalyHandler:
    def __init__(self, 
                 feature_columns: List[str], 
                 bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = None,
                 detect_method: str = 'bounds_and_diff',
                 diff_threshold: float = 3.0,
                 correct_method: str = 'interpolate',
                 phase_shift_pairs: List[Tuple[str, str]] = None,
                 phase_shift_threshold: float = 6.0):
        """
        光伏时间序列np.array的异常值检测与校正模块
        
        :param feature_columns: DataFrame中需要处理的列名（其值为np.array）。包含 history 和 future 列。
        :param bounds: 物理上下界字典，如 {'GHI': (0, 1500), 'TEMP': (-40, 60)}。前缀匹配机制。
        :param detect_method: 检测方法 'bounds_only' (仅物理边界), 'bounds_and_diff' (边界+突变检测), 'zscore' 等
        :param diff_threshold: 若使用突变检测，允许相邻时刻差异超过 std(diff)的倍数（推荐3.0~4.0）
        :param correct_method: 修正方法 'interpolate' (前后向线性插值), 'mean' (均值填充), 'none'
        :param phase_shift_pairs: 用于检测相位偏移的历史和未来列对，如 [('GHI_history', 'GHI_future1d')]
        :param phase_shift_threshold: 相位偏移判定阈值(小时)。因自然光伏主要有午间高峰，质心差异大于此值(默认6小时)视作错位。
        """
        self.feature_columns = feature_columns
        self.bounds = bounds or {}
        self.detect_method = detect_method
        self.diff_threshold = diff_threshold
        self.correct_method = correct_method
        self.phase_shift_pairs = phase_shift_pairs or []
        self.phase_shift_threshold = phase_shift_threshold

    def _detect_anomalies(self, arr: np.ndarray, col_name: str) -> np.ndarray:
        """核心模块：返回一个与目标arr等长的布尔数组，True表示该点是异常值"""
        mask = np.zeros_like(arr, dtype=bool)
        
        # 1. 物理边界检测
        base_name = col_name.split('_')[0] 
        bound_key = col_name if col_name in self.bounds else base_name
            
        if bound_key in self.bounds:
            lower, upper = self.bounds[bound_key]
            if lower is not None:
                mask |= (arr < lower)
            if upper is not None:
                mask |= (arr > upper)
                
        # 2. 突变异常检测
        if 'diff' in self.detect_method:
            diffs = np.abs(np.diff(arr, prepend=arr[0])) 
            std_diff = np.std(diffs)
            if std_diff > 1e-6:
                mask |= (diffs > self.diff_threshold * std_diff) & (diffs > np.max(arr) * 0.1)

        # 3. Z-score 检测
        if self.detect_method == 'zscore':
            std = np.std(arr)
            if std > 1e-6:
                z_scores = np.abs((arr - np.mean(arr)) / std)
                mask |= (z_scores > self.diff_threshold)
                
        return mask

    def _detect_phase_shift(self, hist_arr: np.ndarray, fut_arr: np.ndarray) -> bool:
        """检测未来24小时与历史最后24小时是否发生严重的相位偏移（基于能量质心计算）"""
        if len(hist_arr) < 24 or len(fut_arr) < 24:
            return False
            
        hist_24 = hist_arr[-24:]
        fut_24 = fut_arr[:24]
        
        hist_sum = np.sum(hist_24)
        fut_sum = np.sum(fut_24)
        
        # 如果某一天几乎全是0（极度阴天或夜晚），质心计算无意义，不判定为突发相位偏移
        if hist_sum < 1e-3 or fut_sum < 1e-3:
            return False
            
        # 计算一天内曲线的质心（代表主体能量主要分布在0~23的哪个小时刻度点）
        hist_com = np.sum(np.arange(24) * hist_24) / hist_sum
        fut_com = np.sum(np.arange(24) * fut_24) / fut_sum
        
        # 计算首尾相接的环形时间差（如日均峰值如果在 23点 和 1点，其实际只相距 2 小时）
        diff = abs(hist_com - fut_com)
        diff_circular = min(diff, 24 - diff)
        
        return diff_circular > self.phase_shift_threshold

    def _correct_anomalies(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """核心模块：对布尔数组mask标记为True的地方执行修正"""
        if not np.any(mask):
            return arr
            
        arr_corrected = arr.copy().astype(float)
        
        if self.correct_method == 'interpolate':
            s = pd.Series(arr_corrected)
            s[mask] = np.nan
            s = s.interpolate(method='linear').bfill().ffill()
            arr_corrected = s.values
            
        elif self.correct_method == 'mean':
            valid_mean = np.mean(arr[~mask]) if np.any(~mask) else 0
            arr_corrected[mask] = valid_mean
            
        return arr_corrected

    def process(self, df: pd.DataFrame, do_correct: bool = True) -> pd.DataFrame:
        """
        对传入的DataFrame进行逐行处理
        """
        df_processed = df.copy()
        df_processed['has_anomaly'] = False
        
        anomaly_details = [[] for _ in range(len(df_processed))]

        for col in self.feature_columns:
            if col not in df_processed.columns:
                continue
                
            col_has_anomaly_list = []
            corrected_arrays = []
            
            for idx, arr in enumerate(df_processed[col]):
                mask = self._detect_anomalies(arr, col)
                has_outlier = np.any(mask)
                col_has_anomaly_list.append(has_outlier)
                
                if has_outlier:
                    anomaly_details[idx].append(col)
                
                if has_outlier and do_correct:
                    arr_corrected = self._correct_anomalies(arr, mask)
                    corrected_arrays.append(arr_corrected)
                else:
                    corrected_arrays.append(arr.copy())
                    
            if do_correct:
                df_processed[col] = corrected_arrays
                
            df_processed['has_anomaly'] = df_processed['has_anomaly'] | pd.Series(col_has_anomaly_list)
            
        # 补充：基于时序逻辑对的双端相位偏移检测 (Phase Shift)
        for hist_col, fut_col in self.phase_shift_pairs:
            if hist_col in df_processed.columns and fut_col in df_processed.columns:
                phase_anomaly_list = []
                for idx, (hist_arr, fut_arr) in enumerate(zip(df_processed[hist_col], df_processed[fut_col])):
                    is_shifted = self._detect_phase_shift(hist_arr, fut_arr)
                    phase_anomaly_list.append(is_shifted)
                    if is_shifted:
                        anomaly_details[idx].append(f"PhaseShift({fut_col})")
                        
                df_processed['has_anomaly'] = df_processed['has_anomaly'] | pd.Series(phase_anomaly_list)

        df_processed['anomaly_columns'] = [",".join(cols) if cols else "" for cols in anomaly_details]
            
        return df_processed


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    n_samples = 10
    
    # 模拟含有 np.array 的 DataFrame
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'id': np.random.randint(1, 5, n_samples),
        'pv_data_history': [np.random.rand(168) * 100 for _ in range(n_samples)],
        'pv_data_future1d': [np.random.rand(24) * 100 for _ in range(n_samples)],
        'GHI_history': [np.random.rand(168) * 1000 for _ in range(n_samples)],
        'TEMP_history': [np.random.rand(168) * 30 + 10 for _ in range(n_samples)]
    }
    
    # 手动注入一些异常值
    data['pv_data_history'][0][10] = -50  # 物理边界异常 (负数PV)
    data['GHI_history'][1][50] = 5000     # 物理边界异常 (超大GHI)
    data['TEMP_history'][2][100] = 200    # 物理边界异常 (极高温度)
    # 注入突变异常
    data['pv_data_history'][3][20] = data['pv_data_history'][3][19] + 80
    
    # 注入相位偏移异常 (GHI_future1d 严重错误，平移了 12 小时)
    # 先强制造一个极其标准的光伏钟形日照曲线
    normal_ghi = np.sin(np.linspace(0, np.pi, 24)) * 1000
    normal_ghi = np.clip(normal_ghi, 0, None)
    # 填入正常的 history 最后一天
    data['GHI_history'][4][-24:] = normal_ghi
    # future1d 发生12小时错位 (相当于把中午的太阳强度移到了半夜)
    data['GHI_future1d'][4] = np.roll(normal_ghi, 12) 
    
    df = pd.DataFrame(data)
    
    columns_to_check = [
        'pv_data_history', 'pv_data_future1d', 
        'GHI_history', 'GHI_future1d', 'TEMP_history'
    ]
    
    bounds = {
        'pv_data': (0.0, None),
        'GHI': (0.0, 1500.0),
        'TEMP': (-40.0, 60.0),
    }
    
    handler = TimeSeriesAnomalyHandler(
        feature_columns=columns_to_check,
        bounds=bounds,
        detect_method='bounds_and_diff',
        correct_method='interpolate',
        phase_shift_pairs=[('GHI_history', 'GHI_future1d'), ('pv_data_history', 'pv_data_future1d')],
        phase_shift_threshold=6.0
    )
    
    print("处理前，第一行是否有异常标记：", 'has_anomaly' in df.columns)
    
    # 不校正，仅打标记
    df_analyzed = handler.process(df, do_correct=False)
    print("发现异常的行数 (仅检查):", df_analyzed['has_anomaly'].sum())
    print("包含异常的列详情:")
    print(df_analyzed.loc[df_analyzed['has_anomaly'], ['id', 'anomaly_columns']])
    
    # 检测并校正
    df_corrected = handler.process(df, do_correct=True)
    print("\n校正前第一行旧数据在索引10的值(异常):", df['pv_data_history'][0][10])
    print("校正后新系列在索引10的值(经插值):", df_corrected['pv_data_history'][0][10])
