#!/usr/bin/env python3
"""
RAPIDS Google Colab 环境配置脚本
使用方法：在 Colab Cell 中粘贴以下代码并运行

# === 方法 1：一键配置 ===
!wget -q https://raw.githubusercontent.com/rapidsai/rapids-cuda/branch-23.12/rapids-cloud-config.sh -O- | bash
!pip install cudf-cu12 cuml-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
import cudf, cuml; print(f"cuDF: {cudf.__version__}, cuML: {cuml.__version__}")

# === 方法 2：逐步配置（推荐教学使用） ===
# 见下方代码
"""

import subprocess
import sys
import os


def run(cmd, check=True, capture=False):
    """执行命令并打印输出"""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True,
        capture_output=capture,
        text=True
    )
    if check and result.returncode != 0:
        print(f"  [ERROR] 命令失败 (exit {result.returncode})")
        if capture:
            print(result.stderr)
        return False
    if capture:
        print(result.stdout)
    return True


def check_gpu():
    """Step 1: 检查 GPU 可用性"""
    print("=" * 50)
    print("[STEP 1/5] 检查 GPU 可用性")
    print("=" * 50)
    run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    run("nvcc --version 2>/dev/null || echo 'nvcc 不可用'")


def install_rapids_pip():
    """Step 2: 通过 pip 安装 RAPIDS"""
    print("\n" + "=" * 50)
    print("[STEP 2/5] 安装 RAPIDS 核心库 (pip)")
    print("=" * 50)
    # 检测 CUDA 版本
    try:
        result = subprocess.run(
            "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9]+'",
            shell=True, capture_output=True, text=True
        )
        cuda_major = result.stdout.strip() or "12"
    except Exception:
        cuda_major = "12"

    suffix = f"cu{cuda_major}"
    print(f"  检测到 CUDA {cuda_major}，使用包后缀: -{suffix}")

    packages = [
        f"cudf-{suffix}",
        f"cuml-{suffix}",
        f"cugraph-{suffix}",
    ]
    pip_cmd = f"pip install {' '.join(packages)} --extra-index-url=https://pypi.nvidia.com"
    run(pip_cmd)


def install_aux_libraries():
    """Step 3: 安装辅助库"""
    print("\n" + "=" * 50)
    print("[STEP 3/5] 安装课程辅助库")
    print("=" * 50)
    aux = "xgboost scikit-learn matplotlib seaborn shap"
    run(f"pip install {aux}")


def verify_installation():
    """Step 4: 验证安装"""
    print("\n" + "=" * 50)
    print("[STEP 4/5] 验证 RAPIDS 安装")
    print("=" * 50)
    verify_code = """
import cudf
import cuml
import cugraph

print(f"  cuDF 版本:   {cudf.__version__}")
print(f"  cuML 版本:   {cuml.__version__}")
print(f"  cuGraph 版本: {cugraph.__version__}")

# cuDF 功能测试
s = cudf.Series([1, 2, 3, 4, 5])
assert s.sum() == 15, "cuDF 运算错误"
print("  cuDF 运算测试: ✅")

# cuML 功能测试
from cuml.linear_model import LinearRegression
import numpy as np
X = cudf.DataFrame({'x': np.arange(100).astype(float)})
y = cudf.Series(np.arange(100) * 2.0 + 1)
model = LinearRegression()
model.fit(X, y)
print(f"  回归系数: {model.coef_}, 截距: {model.intercept_}")
print("  cuML 训练测试: ✅")

# XGBoost GPU 测试
import xgboost as xgb
dtrain = xgb.DMatrix(np.random.randn(100, 5), label=np.random.randint(0, 2, 100))
params = {'tree_method': 'hist', 'device': 'cuda', 'objective': 'binary:logistic'}
xgb.train(params, dtrain, num_boost_round=5)
print("  XGBoost GPU 测试: ✅")

print("\\n  🎉 RAPIDS 环境配置成功！")
"""
    run(f'python3 -c "{verify_code}"')


def download_datasets(datadir="./datasets"):
    """Step 5: 下载课程数据集"""
    print("\n" + "=" * 50)
    print("[STEP 5/5] 下载课程实验数据集")
    print("=" * 50)
    os.makedirs(datadir, exist_ok=True)

    datasets_code = f"""
import os, pickle, numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_20newsgroups, fetch_lfw_people, fetch_openml

datadir = "{datadir}"

# 1. California Housing
print("  下载 California Housing...")
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target
df.to_csv(os.path.join(datadir, 'california_housing.csv'), index=False)
print(f"    -> {{df.shape[0]}} 行")

# 2. 20 Newsgroups
print("  下载 20 Newsgroups...")
ng = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
with open(os.path.join(datadir, '20newsgroups.pkl'), 'wb') as f:
    pickle.dump(ng, f)
print(f"    -> {{len(ng.data)}} 篇文档")

# 3. MNIST
print("  下载 MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
np.savez(os.path.join(datadir, 'mnist_784.npz'), X=mnist.data, y=mnist.target.astype(int))
print(f"    -> {{mnist.data.shape[0]}} 样本")

# 4. LFW
print("  下载 LFW...")
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
np.savez(os.path.join(datadir, 'lfw_people.npz'),
         data=lfw.data, target=lfw.target,
         target_names=lfw.target_names, images=lfw.images)
print(f"    -> {{lfw.data.shape[0]}} 样本")

# 5. Adult Income
print("  下载 Adult Income...")
import urllib.request
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
urllib.request.urlretrieve(url, os.path.join(datadir, 'adult.data'))
print(f"    -> adult.data")

print("\\n  ✅ 数据集下载完成！")
print("  注意: Kaggle 数据集需手动配置 API Token 后下载:")
print("    - Credit Card Fraud: kaggle datasets download -d mlg-ulb/creditcardfraud")
print("    - Mall Customers: kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python")
print("    - Home Credit: kaggle competitions download -c home-credit-default-risk")
"""
    run(f'python3 -c "{datasets_code}"')


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  RAPIDS Google Colab 环境配置")
    print("  版本: RAPIDS 26.08 (CUDA 12)")
    print("=" * 50)

    check_gpu()
    install_rapids_pip()
    install_aux_libraries()
    verify_installation()
    download_datasets()

    print("\n" + "=" * 50)
    print("  配置完成！现在可以开始课程实验了。")
    print("=" * 50)
