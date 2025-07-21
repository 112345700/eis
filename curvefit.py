from scipy.optimize import minimize, least_squares
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from matplotlib.font_manager import FontProperties

chinese_font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'
chinese_font = FontProperties(fname=chinese_font_path)
# 数据文件路径和电池类型
data_path = pathlib.Path("Impedance raw data and fitting data")
battery_types = {
    'NCA': 'NCA battery',
    'NCM': 'NCM battery',
    'NCM+NCA': 'NCM+NCA battery'
}

def load_impedance_data(battery_type, file_name, sheet_name=None):
    file_path = pathlib.Path("Downloads/eis-main") / data_path / battery_types[battery_type] / file_name
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return None
    if sheet_name is None:
        xl_file = pd.ExcelFile(file_path, engine="openpyxl")
        return xl_file.sheet_names
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        df.columns = df.columns.map(str)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None

def cpe_impedance(Q, n, w):
    w = np.where(w == 0, 1e-12, w)
    return 1 / (Q * (1j * w) ** n)

def warburg_impedance(W, w):
    w = np.where(w == 0, 1e-12, w)
    return W / np.sqrt(1j * w)

def circuit_impedance(f, R0, R1, Q1, n1, R2, Q2, n2, W):
    w = 2 * np.pi * f
    Z_cpe1 = cpe_impedance(Q1, n1, w)
    Z_R1_cpe1 = 1 / (1/R1 + 1/Z_cpe1)
    Z_warburg = warburg_impedance(W, w)
    Z_R2_w = R2 + Z_warburg
    Z_cpe2 = cpe_impedance(Q2, n2, w)
    Z_R2w_cpe2 = 1 / (1/Z_R2_w + 1/Z_cpe2)
    Z_total = R0 + Z_R1_cpe1 + Z_R2w_cpe2 + Z_warburg
    return np.real(Z_total), np.imag(Z_total)
def filter_positive_imag_and_endpoints(data):
    """过滤虚部为正的数据并删除首尾数据点
    新增: 移除最开始和最后的数据点，curve_fit 无此功能"""
    if "Data: Z''" not in data.columns or "Data: Z'" not in data.columns:
        return data
    # 过滤虚部为正
    data = data[data["Data: Z''"] < 0].reset_index(drop=True)
    # 删除首尾数据点
    if len(data) > 2:  # 确保至少保留两个点
        data = data.iloc[1:-1].reset_index(drop=True)
        print(f"删除了首尾数据点，剩余 {len(data)} 个点")
    else:
        print("数据点不足，未能删除首尾")
    return data

def filter_positive_imag(data):
    """去除虚部为正的阻抗数据"""
    if "Data: Z''" not in data.columns:
        return data
    return data[data["Data: Z''"] < 0].reset_index(drop=True)

def fit_eis_circuit(freq, z_real, z_imag, p0=None, bounds=None, max_nfev=1000, threshold=1e-6,
                   max_retries=10, verbose=False):
    """
    拟合 EIS 数据到ECM R - [R || CPE1] - [(R + W)||CPE2] 电路模型。

    参数:
    - freq: 频率数组 (Hz)
    - z_real: 实部阻抗数组 (Ω)
    - z_imag: 虚部阻抗数组 (Ω)
    - p0: 初始参数数组或列表 [R0, R1, Q1, n1, R2, Q2, n2, W]，若为 None 则自动生成
    - bounds: 参数边界元组 (lb, ub)，其中 lb 和 ub 是长度为 8 的数组 [min, max]
    - max_nfev: 最大函数评估次数
    - threshold: RMSE 收敛阈值
    - max_retries: 最大重试次数
    - verbose: 是否打印调试信息

    返回:
    - popt: 最优参数 [R0, R1, Q1, n1, R2, Q2, n2, W]
    - pcov: 协方差矩阵
    """
    # 合并实部和虚部作为目标值
    y_true = np.concatenate([z_real, z_imag])
    n_points = len(freq)

    # 自动生成初始参数
    if p0 is None:
        z_real_min = np.min(z_real)
        z_real_max = np.max(z_real)
        z_imag_max = np.max(np.abs(z_imag))
        freq_median = np.median(freq[freq > 0])
        freq_min = np.min(freq[freq > 0])
        p0_sets = [
            [z_real_min * 0.9, (z_real_max - z_real_min) * 0.3, 1 / (2 * np.pi * freq_median * z_imag_max), 0.8,
             (z_real_max - z_real_min) * 0.6, 1 / (2 * np.pi * freq_median * z_imag_max), 0.8,
             z_imag_max / np.sqrt(2 * np.pi * freq_min)],
            [z_real_min * 1.5, (z_real_max - z_real_min) * 0.8, 1 / (2 * np.pi * freq_median * z_imag_max * 0.1), 0.95,
             (z_real_max - z_real_min) * 0.2, 1 / (2 * np.pi * freq_median * z_imag_max * 0.1), 0.95,
             z_imag_max / np.sqrt(2 * np.pi * freq_min) * 1.5],
            [z_real_min * 0.7, (z_real_max - z_real_min) * 0.1, 1 / (2 * np.pi * freq_median * z_imag_max * 10), 0.5,
             (z_real_max - z_real_min) * 0.9, 1 / (2 * np.pi * freq_median * z_imag_max * 10), 0.5,
             z_imag_max / np.sqrt(2 * np.pi * freq_min) * 0.5]
        ]
    else:
        p0_sets = [np.array(p0)]

    # 默认边界，格式为 (lb, ub)
    if bounds is None:
        lb = np.array([0, 0, 1e-6, 0, 0, 1e-6, 0, 1e-6])
        ub = np.array([z_real_max * 2, z_real_max * 2, 10, 1, z_real_max * 2, 300, 1, z_imag_max * 10])
        bounds = (lb, ub)
    elif len(bounds) == 2 and len(bounds[0]) == len(bounds[1]) == 8:
        lb, ub = bounds
    else:
        raise ValueError("bounds必须是(lb, ub)格式，其中lb和ub是长度为8的数组")

    def residual(params, freq, z_real, z_imag):
        zr_pred, zi_pred = circuit_impedance(freq, *params)
        y_pred = np.concatenate([zr_pred, zi_pred])
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return y_true - y_pred  # 返回残差

    best_rmse = float('inf')
    best_popt = None
    best_pcov = None

    for p0 in p0_sets:
        current_p0 = np.array(p0.copy())
        for attempt in range(max_retries):
            try:
                # 使用 least_squares 进行拟合 --学习了curve_fit的用法
                result = least_squares(residual, current_p0, args=(freq, z_real, z_imag), bounds=bounds,
                                      max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, verbose=2 if verbose else 0)
                popt = result.x
                zr_pred, zi_pred = circuit_impedance(freq, *popt)
                y_pred = np.concatenate([zr_pred, zi_pred])
                total_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

                if verbose:
                    print(f"✅ p0组 {p0_sets.index(p0)+1}, 第 {attempt+1} 次拟合，总RMSE = {total_rmse:.6f}, "
                          f"参数={popt}")

                # 计算协方差矩阵（近似）
                pcov = np.zeros((len(popt), len(popt)))
                if result.cost > 0 and len(y_true) > len(popt):
                    s_sq = result.cost / (len(y_true) - len(popt))
                    pcov = s_sq * np.linalg.inv(result.jac.T.dot(result.jac))

                if total_rmse < best_rmse:
                    best_rmse = total_rmse
                    best_popt = popt
                    best_pcov = pcov

                if total_rmse < threshold:
                    return best_popt, best_pcov

                # 退火式调整初始值
                if total_rmse > threshold:
                    current_p0 = popt * (1 + np.random.uniform(-0.2, 0.2, size=len(popt)))
                    current_p0 = np.clip(current_p0, lb, ub)

            except Exception as e:
                if verbose:
                    print(f"❌ p0组 {p0_sets.index(p0)+1}, 第 {attempt+1} 次拟合失败：{e}")
                current_p0 = popt * (1 + np.random.uniform(-0.2, 0.2, size=len(popt)))
                current_p0 = np.clip(current_p0, lb, ub)

    if best_popt is None:
        raise ValueError("拟合失败，未找到可接受的参数")
    return best_popt, best_pcov

def plot_fit_comparison(data, popt):
    freq = data["Data: Frequency"].values
    z_real = data["Data: Z'"].values
    z_imag = data["Data: Z''"].values
    my_zr, my_zi = circuit_impedance(freq, *popt)

    plt.figure(figsize=(8, 6))
    if "Data: Z'" in data.columns and "Data: Z''" in data.columns:
        plt.scatter(z_real, -z_imag, alpha=0.7, s=30, label="原始数据", color="black")
    plt.tight_layout()
    plt.plot(my_zr, -my_zi, label="拟合曲线", color="blue")
    plt.xlabel("Z' (Ω)", fontproperties=chinese_font)
    plt.ylabel("-Z'' (Ω)", fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    plt.title("Nyquist 拟合对比", fontproperties=chinese_font)
    plt.axis('equal')
    plt.show()

# 示例：加载NCA电池的第一个工作表并拟合
battery_type = 'NCM+NCA'  # 可选 'NCA', 'NCM', 'NCM+NCA'
file_name = 'CY25_0.5_1.xlsx'
sheets = load_impedance_data(battery_type, file_name)
all_fit_results = []
if sheets:
    for sheet in sheets:
        print(f"\n正在处理工作表: {sheet}")
        data = load_impedance_data(battery_type, file_name, sheet)
        if data is not None:
            data = filter_positive_imag_and_endpoints(data)
            if data.empty:
                print("没有有效数据，跳过此工作表")# 实际上我们的数据组不需要
                continue
            try:
                # 使用 fit_eis_circuit 替代 auto_fit_with_retry
                popt, pcov = fit_eis_circuit(
                    data["Data: Frequency"].values,
                    data["Data: Z'"].values,
                    data["Data: Z''"].values,
                    p0=None, #参数已经被指定
                    bounds=None,  # 自动生成边界 (lb, ub)
                    max_nfev=1000,  # 最大循环次数
                    threshold=0.00001,  # RMSE 阈值
                    max_retries=15,  # 最大重试次数（效果不好的基础上）
                    verbose=True  # 调试输出（true表示开启日志，可以开可以不开）
                )
                plot_fit_comparison(data, popt)

                result_with_sheet = {'sheet': sheet}
                param_names = ["R0", "R1", "Q1", "n1", "R2", "Q2", "n2", "W"]
                result_with_sheet.update(dict(zip(param_names, popt)))
                all_fit_results.append(result_with_sheet)

                print("拟合参数:", dict(zip(param_names, popt)))
                print("协方差矩阵:", pcov)
            except Exception as e:
                print(f"拟合失败: {e}")
        else:
            print("数据加载失败")

    result_df = pd.DataFrame(all_fit_results)
    output_path = pathlib.Path("Downloads/eis-main") / f"{battery_type}{file_name}_fitting_results.xlsx"
    result_df.to_excel(output_path, index=False)
    print(f"\n所有拟合参数已保存到: {output_path}")