"""光伏功率预测数据预处理模块 - 异常值检测与校正"""

##我现在正在进行一个光伏功率预测任务。我的数据形式是这样的dataframe, 其中timestampl列表示时间戳，id表示站点的station id,                                
#pv_data_history列中每个值都是np.array，长度都是7*24，表示历史7天的pv序列，以1h为分辨率，pv_data_future1d列中每个值都是np.array，                        
#长度都是1*24,表示未来1天的pv序列，以1h问分辨率。df中还有与气象要素相关的列，例如GHI和TEMP。举例对GHI, GHI_history这一列中每个值都是np.array，长度为7*24 
#，表示历史7天的GHI,GHI_future1d中每个值都是np.array，长度为1*24，表示未来1天的天气预报GHI。其他气象要素的数据同理。我有一个深度学习预测模型，输入的是除 
#pv_data_future1d的所有列，然后输出的预测值和pv_data_future1d来进行指标计算。我在回测阶段发现数据有这样的一些问题。首先就是可能会在这些np.array中出现极  
#端异常值，我需要一个数据预处理模块，对这样有异常值的行进行标记，以及可以选择是否对异常值进行校正。帮我构造这样的方法python代码  
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class AnomalyDetectionConfig:
    """异常值检测配置"""
    # 检测方法: 'iqr', 'zscore', 'percentile', 'physical'
    method: str = 'iqr'

    # IQR方法参数
    iqr_multiplier: float = 3.0  # IQR倍数，用于确定异常值边界

    # Z-score方法参数
    zscore_threshold: float = 3.0

    # 百分位方法参数
    percentile_low: float = 1.0   # 下界百分位
    percentile_high: float = 99.0  # 上界百分位

    # 物理限制方法参数
    physical_min: Optional[float] = None  # 最小物理值（如功率 >= 0）
    physical_max: Optional[float] = None  # 最大物理值（如功率 <= 装机容量）

    # 是否检查负值（对于光伏功率应该为True）
    check_negative: bool = True


@dataclass
class AnomalyCorrectionConfig:
    """异常值校正配置"""
    # 校正方法: 'clip', 'interpolate', 'mean', 'median', 'nan'
    method: str = 'clip'

    # clip方法参数
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    # interpolate方法参数
    interpolate_limit: int = 24  # 连续异常值最大插值数量                   


class OutlierDetector:
    """光伏数据异常值检测器

    支持多种检测方法，可对包含np.array的DataFrame进行异常值检测和标记。
    """

    # 默认需要检查的np.array列
    DEFAULT_ARRAY_COLUMNS = [
        'pv_data_history', 'pv_data_future1d',
        'GHI_history', 'GHI_future1d',
        'TEMP_history', 'TEMP_future1d',
        # 可根据实际情况添加更多
    ]

    def __init__(
        self,
        detection_config: Optional[AnomalyDetectionConfig] = None,
        correction_config: Optional[AnomalyCorrectionConfig] = None,
    ):
        """
        Args:
            detection_config: 异常值检测配置
            correction_config: 异常值校正配置
        """
        self.detection_config = detection_config or AnomalyDetectionConfig()
        self.correction_config = correction_config or AnomalyCorrectionConfig()

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        array_columns: Optional[List[str]] = None,
        return_mask: bool = True,
    ) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame]]:
        """检测异常值

        Args:
            df: 输入DataFrame
            array_columns: 需要检查的np.array列名列表
            return_mask: 是否返回布尔掩码（True返回掩码DataFrame，False返回标记列）

        Returns:
            如果return_mask=True: 返回两个DataFrame - (原始df添加标记列, 异常值掩码df)
            否则: 返回添加了'anomaly_mask'列的DataFrame
        """
        if array_columns is None:
            array_columns = self._detect_array_columns(df)

        # 初始化标记列
        anomaly_mask_df = pd.DataFrame(index=df.index)
        label_df = pd.DataFrame(index=df.index)

        # 对每一列进行异常值检测
        for col in array_columns:
            if col not in df.columns:
                continue

            col_mask = df[col].apply(lambda x: self._detect_single_array(x))
            anomaly_mask_df[f'{col}_is_anomaly'] = col_mask
            label_df[f'{col}_anomaly'] = col_mask

        # 汇总：任意列有异常值即为异常
        overall_mask = anomaly_mask_df.any(axis=1)
        label_df['has_anomaly'] = overall_mask

        if return_mask:
            # 添加标记列到原df
            result_df = df.copy()
            for col in label_df.columns:
                result_df[col] = label_df[col].values
            return result_df, anomaly_mask_df
        else:
            result_df = df.copy()
            result_df['has_anomaly'] = overall_mask.values
            return result_df

    def _detect_single_array(self, arr: np.ndarray) -> bool:
        """检测单个数组是否有异常值"""
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # 移除NaN进行计算
        valid_data = arr[~np.isnan(arr)]
        if len(valid_data) == 0:
            return False

        method = self.detection_config.method

        if method == 'iqr':
            return self._detect_iqr(valid_data)
        elif method == 'zscore':
            return self._detect_zscore(valid_data)
        elif method == 'percentile':
            return self._detect_percentile(valid_data)
        elif method == 'physical':
            return self._detect_physical(valid_data)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _detect_iqr(self, data: np.ndarray) -> bool:
        """IQR方法检测异常值"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - self.detection_config.iqr_multiplier * iqr
        upper = q3 + self.detection_config.iqr_multiplier * iqr
        return np.any((data < lower) | (data > upper))

    def _detect_zscore(self, data: np.ndarray) -> bool:
        """Z-score方法检测异常值"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return False
        z_scores = np.abs((data - mean) / std)
        return np.any(z_scores > self.detection_config.zscore_threshold)

    def _detect_percentile(self, data: np.ndarray) -> bool:
        """百分位方法检测异常值"""
        lower = np.percentile(data, self.detection_config.percentile_low)
        upper = np.percentile(data, self.detection_config.percentile_high)
        return np.any((data < lower) | (data > upper))

    def _detect_physical(self, data: np.ndarray) -> bool:
        """物理限制方法检测异常值"""
        config = self.detection_config
        has_anomaly = False

        # 检查负值
        if config.check_negative and np.any(data < 0):
            has_anomaly = True

        # 检查物理最小值
        if config.physical_min is not None and np.any(data < config.physical_min):
            has_anomaly = True

        # 检查物理最大值
        if config.physical_max is not None and np.any(data > config.physical_max):
            has_anomaly = True

        return has_anomaly

    def correct_anomalies(
        self,
        df: pd.DataFrame,
        array_columns: Optional[List[str]] = None,
        correction_config: Optional[AnomalyCorrectionConfig] = None,
    ) -> pd.DataFrame:
        """校正异常值

        Args:
            df: 输入DataFrame
            array_columns: 需要校正的np.array列名列表
            correction_config: 校正配置，如果为None使用实例的配置

        Returns:
            校正后的DataFrame
        """
        if array_columns is None:
            array_columns = self._detect_array_columns(df)

        config = correction_config or self.correction_config

        result_df = df.copy()

        for col in array_columns:
            if col not in df.columns:
                continue
            result_df[col] = result_df[col].apply(lambda x: self._correct_single_array(x, config))

        return result_df

    def _correct_single_array(
        self,
        arr: np.ndarray,
        config: AnomalyCorrectionConfig
    ) -> np.ndarray:
        """校正单个数组的异常值"""
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        result = arr.copy()
        method = config.method

        if method == 'clip':
            result = self._clip_array(result, config)
        elif method == 'interpolate':
            result = self._interpolate_array(result, config)
        elif method == 'mean':
            result = self._fill_mean(result)
        elif method == 'median':
            result = self._fill_median(result)
        elif method == 'nan':
            pass  # 保持NaN不变
        else:
            raise ValueError(f"Unknown correction method: {method}")

        return result

    def _clip_array(self, arr: np.ndarray, config: AnomalyCorrectionConfig) -> np.ndarray:
        """裁剪数组到合理范围"""
        result = arr.copy()

        # 使用配置的范围
        clip_min = config.clip_min
        clip_max = config.clip_max

        # 如果没有指定范围，使用IQR自动确定
        if clip_min is None or clip_max is None:
            valid_data = result[~np.isnan(result)]
            if len(valid_data) > 0:
                q1 = np.percentile(valid_data, 25)
                q3 = np.percentile(valid_data, 75)
                iqr = q3 - q1
                if clip_min is None:
                    clip_min = q1 - 3 * iqr
                if clip_max is None:
                    clip_max = q3 + 3 * iqr

        if clip_min is not None:
            result = np.maximum(result, clip_min)
        if clip_max is not None:
            result = np.minimum(result, clip_max)

        return result

    def _interpolate_array(self, arr: np.ndarray, config: AnomalyCorrectionConfig) -> np.ndarray:
        """使用插值替换异常值"""
        result = arr.copy()

        # 先标记异常值位置
        valid_data = result[~np.isnan(result)]
        if len(valid_data) == 0:
            return result

        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr

        # 创建掩码：异常值为True
        anomaly_mask = (result < lower) | (result > upper)

        if not np.any(anomaly_mask):
            return result

        # 用插值替换异常值
        result[anomaly_mask] = np.interp(
            np.where(anomaly_mask)[0],
            np.where(~anomaly_mask)[0],
            result[~anomaly_mask]
        )

        return result

    def _fill_mean(self, arr: np.ndarray) -> np.ndarray:
        """用均值替换异常值"""
        result = arr.copy()
        valid_data = result[~np.isnan(result)]
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            mask = (result < lower) | (result > upper)
            result[mask] = mean_val
        return result

    def _fill_median(self, arr: np.ndarray) -> np.ndarray:
        """用中位数替换异常值"""
        result = arr.copy()
        valid_data = result[~np.isnan(result)]
        if len(valid_data) > 0:
            median_val = np.median(valid_data)
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            mask = (result < lower) | (result > upper)
            result[mask] = median_val
        return result

    def _detect_array_columns(self, df: pd.DataFrame) -> List[str]:
        """自动检测包含np.array的列"""
        array_columns = []
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].iloc[0]
                if isinstance(sample, np.ndarray):
                    array_columns.append(col)
        return array_columns


class PVDataPreprocessor:
    """光伏数据综合预处理器

    整合异常值检测、标记和校正功能。
    """

    def __init__(
        self,
        max_power: Optional[float] = None,  # 装机容量，用于物理限制
        enable_correction: bool = False,    # 是否启用校正
        detection_method: str = 'iqr',
        correction_method: str = 'clip',
    ):
        """
        Args:
            max_power: 最大功率（装机容量），用于物理限制
            enable_correction: 是否启用异常值校正
            detection_method: 异常值检测方法
            correction_method: 异常值校正方法
        """
        self.max_power = max_power
        self.enable_correction = enable_correction

        # 配置检测器
        self.detector = OutlierDetector(
            detection_config=AnomalyDetectionConfig(
                method=detection_method,
                physical_max=max_power,
                check_negative=True,
            ),
            correction_config=AnomalyCorrectionConfig(
                method=correction_method,
                clip_max=max_power if max_power else None,
            )
        )

    def process(
        self,
        df: pd.DataFrame,
        array_columns: Optional[List[str]] = None,
        mark_only: bool = False,
    ) -> pd.DataFrame:
        """处理数据

        Args:
            df: 输入DataFrame
            array_columns: 需要处理的np.array列
            mark_only: 如果为True，只标记不校正

        Returns:
            处理后的DataFrame，包含异常值标记列
        """
        # 检测异常值并标记
        result_df, anomaly_mask = self.detector.detect_anomalies(
            df, array_columns=array_columns
        )

        # 统计信息
        n_anomaly = result_df['has_anomaly'].sum()
        print(f"检测到 {n_anomaly} 行包含异常值 ({n_anomaly/len(df)*100:.2f}%)")

        # 详细统计每个列的异常情况
        for col in anomaly_mask.columns:
            n_col_anomaly = anomaly_mask[col].sum()
            if n_col_anomaly > 0:
                print(f"  - {col}: {n_col_anomaly} 行")

        # 校正异常值
        if self.enable_correction and not mark_only:
            result_df = self.detector.correct_anomalies(
                result_df, array_columns=array_columns
            )
            print("异常值已校正")

        return result_df


def create_anomaly_report(
    df: pd.DataFrame,
    array_columns: List[str],
) -> Dict:
    """生成异常值报告

    Args:
        df: 包含异常值标记的DataFrame
        array_columns: 需要检查的列

    Returns:
        包含统计信息的字典
    """
    report = {}

    # 统计总体异常
    if 'has_anomaly' in df.columns:
        report['total_anomalies'] = int(df['has_anomaly'].sum())
        report['anomaly_rate'] = float(df['has_anomaly'].mean())

    # 统计各列异常
    col_reports = {}
    for col in array_columns:
        anomaly_col = f'{col}_is_anomaly'
        if anomaly_col in df.columns:
            col_reports[col] = {
                'count': int(df[anomaly_col].sum()),
                'rate': float(df[anomaly_col].mean())
            }
    report['column_anomalies'] = col_reports

    return report


# ============ 示例用法 ============

if __name__ == '__main__':
    # 创建示例数据
    np.random.seed(42)

    # 模拟正常数据
    n_samples = 100
    normal_data = np.random.randn(n_samples, 168) * 10 + 50  # 均值50，标准差10
    normal_data = np.clip(normal_data, 0, None)  # 确保非负

    # 注入异常值
    anomaly_data = normal_data.copy()
    anomaly_data[5, :10] = 500  # 极端高值
    anomaly_data[10, :5] = -50   # 负值
    anomaly_data[20, 50:60] = 200  # 局部高值

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'id': np.random.randint(1, 5, n_samples),
        'pv_data_history': [arr for arr in anomaly_data],
        'pv_data_future1d': [np.random.rand(24) * 50 for _ in range(n_samples)],
        'GHI_history': [np.random.rand(168) * 800 + 200 for _ in range(n_samples)],
        'GHI_future1d': [np.random.rand(24) * 800 + 200 for _ in range(n_samples)],
    })

    print("=" * 60)
    print("示例1: 使用IQR方法检测并标记异常值")
    print("=" * 60)

    # 创建预处理器（不校正，只标记）
    preprocessor = PVDataPreprocessor(
        max_power=150,  # 假设装机容量150
        enable_correction=False,
        detection_method='iqr',
    )

    result_df = preprocessor.process(df.copy())

    # 生成报告
    report = create_anomaly_report(
        result_df,
        ['pv_data_history', 'pv_data_future1d', 'GHI_history', 'GHI_future1d']
    )
    print("\n异常值报告:")
    print(f"  总异常行数: {report['total_anomalies']}")
    print(f"  异常比例: {report['anomaly_rate']*100:.2f}%")
    for col, stats in report['column_anomalies'].items():
        print(f"  {col}: {stats['count']} 个异常")

    print("\n" + "=" * 60)
    print("示例2: 检测并校正异常值")
    print("=" * 60)

    # 创建预处理器（检测+校正）
    preprocessor_correct = PVDataPreprocessor(
        max_power=150,
        enable_correction=True,
        detection_method='iqr',
        correction_method='clip',
    )

    result_df_corrected = preprocessor_correct.process(df.copy())

    # 对比校正前后的数据
    print(f"\n原始数据[5, :10]: {df['pv_data_history'].iloc[5, :10]}")
    print(f"校正后数据[5, :10]: {result_df_corrected['pv_data_history'].iloc[5, :10]}")

    print("\n" + "=" * 60)
    print("示例3: 使用物理限制方法")
    print("=" * 60)

    detector = OutlierDetector(
        detection_config=AnomalyDetectionConfig(
            method='physical',
            physical_min=0,
            physical_max=150,
            check_negative=True,
        )
    )

    result_df_physical, mask_df = detector.detect_anomalies(df.copy())
    n_physical_anomaly = result_df_physical['has_anomaly'].sum()
    print(f"物理限制方法检测到 {n_physical_anomaly} 行有异常")
