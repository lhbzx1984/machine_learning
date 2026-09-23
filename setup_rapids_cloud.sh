#!/usr/bin/env bash
# ============================================================================
# RAPIDS 云端环境配置脚本 (v26.08)
# 适用平台：Google Colab / Amazon SageMaker / Azure ML / 本地 Conda / Docker
# 用法：bash setup_rapids_cloud.sh [platform]
#   platform 可选: colab | sagemaker | azure | conda | docker | verify | datasets
# 示例：bash setup_rapids_cloud.sh colab
# ============================================================================

set -euo pipefail

# ---- 颜色输出 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
step()    { echo -e "${BLUE}[STEP]${NC} $1"; }

# ---- 版本配置 ----
RAPIDS_VERSION="26.08"
PYTHON_VERSION="3.12"
CUDA_VERSION_CONSTRAINT="cuda-version>=12.0,<=12.9"

# ---- 帮助信息 ----
usage() {
    cat <<EOF
$(echo -e ${GREEN}RAPIDS 云端环境配置脚本${NC})
$(echo -e ${BLUE}版本: RAPIDS ${RAPIDS_VERSION}${NC})

用法: bash $0 <platform>

可用平台:
  colab       Google Colab 环境 (pip 安装)
  sagemaker   Amazon SageMaker Studio Lab (conda 安装)
  azure       Azure Machine Learning Compute Instance (conda 安装)
  conda       本地 Conda 环境 (Miniforge + conda 安装)
  docker      Docker 容器部署 (官方镜像)
  verify      验证已安装的 RAPIDS 环境
  datasets    下载课程实验所需数据集
  all         依次执行: 安装 + 验证 + 数据集下载

示例:
  bash $0 colab
  bash $0 conda
  bash $0 verify
  bash $0 datasets
EOF
    exit 0
}

# ============================================================================
# 平台 1: Google Colab (pip 安装)
# ============================================================================
setup_colab() {
    info "=== Google Colab RAPIDS 环境配置 ==="
    warn "请确保已在 Colab 中启用 GPU: 修改 -> 笔记本设置 -> 硬件加速器 -> T4 GPU"

    # Step 1: 检查 GPU
    step "1/5 检查 GPU 可用性"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
        error "未检测到 GPU！请在 Colab 设置中启用 GPU 加速器。"
        exit 1
    }

    # Step 2: 检查 CUDA 版本
    step "2/5 检查 CUDA 版本"
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "12.2")
    info "检测到 CUDA 版本: ${CUDA_VERSION}"
    CUDA_SUFFIX="cu12"
    if [[ "${CUDA_VERSION%.*}" -ge 13 ]]; then
        CUDA_SUFFIX="cu13"
    fi
    info "使用包后缀: -${CUDA_SUFFIX}"

    # Step 3: 安装 RAPIDS pip 包
    step "3/5 安装 RAPIDS 核心库 (pip)"
    pip install \
        cudf-${CUDA_SUFFIX} \
        cuml-${CUDA_SUFFIX} \
        cugraph-${CUDA_SUFFIX} \
        dask-cudf-${CUDA_SUFFIX} \
        --extra-index-url=https://pypi.nvidia.com

    # Step 4: 安装辅助库
    step "4/5 安装课程辅助库"
    pip install \
        xgboost \
        scikit-learn \
        matplotlib \
        seaborn \
        shap \
        jupyterlab

    # Step 5: 验证
    step "5/5 验证安装"
    python3 -c "
import cudf
import cuml
print(f'cuDF 版本: {cudf.__version__}')
print(f'cuML 版本: {cuml.__version__}')
s = cudf.Series([1, 2, 3, 4, 5])
print(f'cuDF 测试: {s.sum()}')
print('✅ RAPIDS Colab 环境配置成功！')
"

    info "Colab 环境配置完成！"
    info "验证命令: python3 -c 'import cudf; print(cudf.__version__)'"
}

# ============================================================================
# 平台 2: Amazon SageMaker Studio Lab (conda 安装)
# ============================================================================
setup_sagemaker() {
    info "=== Amazon SageMaker Studio Lab RAPIDS 环境配置 ==="
    warn "请确保已申请 SageMaker Studio Lab GPU 账户"

    # Step 1: 检查 conda
    step "1/4 检查 Conda 环境"
    if ! command -v conda &> /dev/null; then
        error "未检测到 conda。SageMaker Studio Lab 应自带 conda。"
        exit 1
    fi
    info "Conda 版本: $(conda --version)"

    # Step 2: 创建 RAPIDS conda 环境
    step "2/4 创建 RAPIDS conda 环境"
    conda create -y -n rapids \
        -c rapidsai -c conda-forge \
        rapids=${RAPIDS_VERSION} \
        python=${PYTHON_VERSION} \
        "${CUDA_VERSION_CONSTRAINT}"

    # Step 3: 激活环境并安装辅助库
    step "3/4 安装课程辅助库"
    eval "$(conda shell.bash hook)"
    conda activate rapids

    pip install xgboost shap scikit-learn matplotlib seaborn

    # Step 4: 注册 Jupyter 内核
    step "4/4 注册 Jupyter 内核"
    python -m ipykernel install --user --name rapids --display-name "RAPIDS ${RAPIDS_VERSION}"

    info "SageMaker 环境配置完成！"
    info "在 JupyterLab 中选择内核 'RAPIDS ${RAPIDS_VERSION}' 即可使用。"
}

# ============================================================================
# 平台 3: Azure Machine Learning Compute Instance (conda 安装)
# ============================================================================
setup_azure() {
    info "=== Azure Machine Learning RAPIDS 环境配置 ==="
    warn "请在 Azure ML Studio 中创建 GPU Compute Instance (如 Standard_NC6s_v3)"
    warn "将此脚本作为创建脚本 (Provision with a creation script) 上传"

    # Step 1: 检查环境
    step "1/5 检查 Azure ML 环境"
    if id -u azureuser &>/dev/null; then
        info "检测到 azureuser 用户"
    else
        warn "未检测到 azureuser，将使用当前用户"
    fi

    # Step 2: 初始化 conda
    step "2/5 初始化 Conda"
    if [ -f /anaconda/etc/profile.d/conda.sh ]; then
        source /anaconda/etc/profile.d/conda.sh
    else
        eval "$(conda shell.bash hook)"
    fi

    # Step 3: 创建 RAPIDS 环境
    step "3/5 创建 RAPIDS conda 环境"
    conda create -y -n rapids \
        --override-channels \
        -c rapidsai -c conda-forge \
        -c microsoft \
        rapids=${RAPIDS_VERSION} \
        python=${PYTHON_VERSION} \
        "${CUDA_VERSION_CONSTRAINT}" \
        'azure-identity>=1.19' \
        ipykernel

    # Step 4: 安装辅助库
    step "4/5 安装课程辅助库"
    conda activate rapids
    pip install \
        'azure-ai-ml>=1.24' \
        xgboost shap scikit-learn matplotlib seaborn

    # Step 5: 注册内核
    step "5/5 注册 Jupyter 内核"
    python -m ipykernel install --user --name rapids --env CONDA_PREFIX "$CONDA_PREFIX"
    echo "kernel install completed"

    info "Azure ML 环境配置完成！"
    info "在 JupyterLab 中选择内核 'rapids' 即可使用。"
}

# ============================================================================
# 平台 4: 本地 Conda 环境 (Miniforge)
# ============================================================================
setup_conda() {
    info "=== 本地 Conda 环境 RAPIDS 配置 ==="

    # Step 1: 检查/安装 Miniforge
    step "1/5 检查 Conda 环境"
    if ! command -v conda &> /dev/null; then
        warn "未检测到 conda，开始安装 Miniforge..."
        step "  下载 Miniforge"
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        step "  安装 Miniforge"
        bash "Miniforge3-$(uname)-$(uname -m).sh" -b
        eval "$HOME/miniforge3/bin/conda" init
        eval "$HOME/miniforge3/bin/conda" config --set channel_priority flexible
        warn "请重启终端后再次运行此脚本，或执行: source ~/.bashrc"
        exit 0
    fi
    info "Conda 版本: $(conda --version)"

    # Step 2: 检查 GPU 和 CUDA
    step "2/5 检查 GPU 和 CUDA 驱动"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        error "未检测到 NVIDIA GPU 驱动！RAPIDS 需要 NVIDIA GPU (Volta 架构及以上)。"
        exit 1
    fi

    # Step 3: 设置 conda channel
    step "3/5 配置 Conda Channel"
    conda config --set channel_priority flexible
    info "Channel priority: flexible"

    # Step 4: 创建 RAPIDS 环境
    step "4/5 创建 RAPIDS conda 环境"
    conda create -y -n rapids-${RAPIDS_VERSION} \
        -c rapidsai -c conda-forge \
        rapids=${RAPIDS_VERSION} \
        python=${PYTHON_VERSION} \
        "${CUDA_VERSION_CONSTRAINT}"

    # Step 5: 安装辅助库
    step "5/5 安装课程辅助库"
    eval "$(conda shell.bash hook)"
    conda activate rapids-${RAPIDS_VERSION}
    pip install xgboost shap scikit-learn matplotlib seaborn jupyterlab

    info "本地 Conda 环境配置完成！"
    info "激活环境: conda activate rapids-${RAPIDS_VERSION}"
    info "启动 Jupyter: jupyter lab"
}

# ============================================================================
# 平台 5: Docker 容器部署
# ============================================================================
setup_docker() {
    info "=== Docker 容器 RAPIDS 部署 ==="

    # Step 1: 检查 Docker
    step "1/5 检查 Docker 环境"
    if ! command -v docker &> /dev/null; then
        warn "未检测到 Docker，开始安装..."
        curl https://get.docker.com | sh
        sudo service docker start
    fi
    info "Docker 版本: $(docker --version)"

    # Step 2: 检查 NVIDIA Container Toolkit
    step "2/5 检查 NVIDIA Container Toolkit"
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        warn "未检测到 NVIDIA Container Toolkit，请按以下步骤安装："
        cat <<EOF
  # Ubuntu/Debian:
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
EOF
        exit 1
    fi
    info "NVIDIA Container Toolkit 已安装"

    # Step 3: 测试 GPU 支持
    step "3/5 测试 Docker GPU 支持"
    docker run --gpus all --rm nvcr.io/nvidia/k8s/cuda-sample:nbody \
        nbody -gpu -benchmark 2>/dev/null && info "GPU 测试通过" || warn "GPU 测试失败，请检查驱动版本"

    # Step 4: 拉取 RAPIDS 镜像
    step "4/5 拉取 RAPIDS Docker 镜像"
    local IMAGE="rapidsai/base:${RAPIDS_VERSION}-cuda12-py3.13"
    info "拉取镜像: ${IMAGE}"
    docker pull "${IMAGE}"

    # Step 5: 输出运行命令
    step "5/5 启动容器"
    cat <<EOF
${GREEN}镜像拉取完成！${NC}

启动交互式容器:
  docker run -it --gpus all ${IMAGE} /bin/bash

启动 JupyterLab (端口 8888):
  docker run -it --gpus all -p 8888:8888 ${IMAGE} \\
    bash -c "jupyter lab --ip=0.0.0.0 --allow-root"

挂载本地目录:
  docker run -it --gpus all -v \$PWD:/workspace ${IMAGE} /bin/bash

多 GPU 分布式训练:
  docker run -t -d --gpus all \\
    --shm-size=1g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    -v \$PWD:/workspace \\
    ${IMAGE}
EOF
}

# ============================================================================
# 验证已安装的 RAPIDS 环境
# ============================================================================
verify_rapids() {
    info "=== RAPIDS 环境验证 ==="

    step "1/6 检查 GPU 硬件"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
        error "未检测到 GPU！"
        exit 1
    }

    step "2/6 检查 CUDA 版本"
    nvcc --version 2>/dev/null || warn "nvcc 不可用（conda 安装时正常）"

    step "3/6 验证 cuDF"
    python3 -c "
import cudf
print(f'  cuDF 版本: {cudf.__version__}')
df = cudf.DataFrame({'a': [1,2,3,4,5], 'b': [10,20,30,40,50]})
print(f'  cuDF 运算测试: {df[\"a\"].sum()}, {df[\"b\"].mean()}')
print('  ✅ cuDF 正常')
" 2>/dev/null || error "cuDF 验证失败"

    step "4/6 验证 cuML"
    python3 -c "
import cuml
print(f'  cuML 版本: {cuml.__version__}')
from cuml.linear_model import LinearRegression
import cudf, numpy as np
X = cudf.DataFrame({'x': np.arange(100).astype(float)})
y = cudf.Series(np.arange(100) * 2.0 + 1)
model = LinearRegression()
model.fit(X, y)
print(f'  回归系数: {model.coef_}, 截距: {model.intercept_}')
print('  ✅ cuML 正常')
" 2>/dev/null || error "cuML 验证失败"

    step "5/6 验证 cuGraph"
    python3 -c "
import cugraph
print(f'  cuGraph 版本: {cugraph.__version__}')
print('  ✅ cuGraph 正常')
" 2>/dev/null || warn "cuGraph 不可用（可选组件）"

    step "6/6 验证 XGBoost GPU"
    python3 -c "
import xgboost as xgb
print(f'  XGBoost 版本: {xgb.__version__}')
import numpy as np
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
dtrain = xgb.DMatrix(X, label=y)
params = {'tree_method': 'hist', 'device': 'cuda', 'objective': 'binary:logistic'}
model = xgb.train(params, dtrain, num_boost_round=10)
print('  ✅ XGBoost GPU 正常')
" 2>/dev/null || warn "XGBoost GPU 验证失败"

    echo ""
    info "========== 验证完成 =========="
    info "如所有组件显示 ✅，环境已就绪！"
}

# ============================================================================
# 数据集下载
# ============================================================================
download_datasets() {
    info "=== 课程实验数据集下载 ==="
    local DATADIR="${1:-./datasets}"
    mkdir -p "${DATADIR}"
    info "数据集将下载到: ${DATADIR}/"

    # ---- 1. California Housing (sklearn API) ----
    step "1/9 California Housing Dataset"
    python3 -c "
from sklearn.datasets import fetch_california_housing
import pandas as pd
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target
df.to_csv('${DATADIR}/california_housing.csv', index=False)
print(f'  下载完成: {df.shape[0]} 行, {df.shape[1]} 列')
" 2>/dev/null && info "  -> ${DATADIR}/california_housing.csv" || warn "  California Housing 下载失败"

    # ---- 2. Credit Card Fraud (Kaggle API) ----
    step "2/9 Credit Card Fraud Detection (Kaggle)"
    if command -v kaggle &> /dev/null; then
        kaggle datasets download -d mlg-ulb/creditcardfraud -p "${DATADIR}" --unzip 2>/dev/null \
            && info "  -> ${DATADIR}/creditcard.csv" \
            || warn "  Kaggle 下载失败，请检查 API Token 配置"
    else
        warn "  未安装 Kaggle CLI。安装: pip install kaggle"
        warn "  配置: 将 kaggle.json 放入 ~/.kaggle/"
        warn "  手动下载: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    fi

    # ---- 3. 20 Newsgroups (sklearn API) ----
    step "3/9 20 Newsgroups Dataset"
    python3 -c "
from sklearn.datasets import fetch_20newsgroups
import pickle
newsgroups = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
with open('${DATADIR}/20newsgroups.pkl', 'wb') as f:
    pickle.dump(newsgroups, f)
print(f'  下载完成: {len(newsgroups.data)} 篇文档, {len(newsgroups.target_names)} 个类别')
" 2>/dev/null && info "  -> ${DATADIR}/20newsgroups.pkl" || warn "  20 Newsgroups 下载失败"

    # ---- 4. MNIST (sklearn fetch_openml) ----
    step "4/9 MNIST Dataset"
    python3 -c "
from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
np.savez('${DATADIR}/mnist_784.npz', X=X, y=y)
print(f'  下载完成: {X.shape[0]} 样本, {X.shape[1]} 维特征')
" 2>/dev/null && info "  -> ${DATADIR}/mnist_784.npz" || warn "  MNIST 下载失败"

    # ---- 5. Labeled Faces in the Wild (sklearn API) ----
    step "5/9 Labeled Faces in the Wild (LFW)"
    python3 -c "
from sklearn.datasets import fetch_lfw_people
import numpy as np
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
np.savez('${DATADIR}/lfw_people.npz',
         data=lfw.data, target=lfw.target,
         target_names=lfw.target_names,
         images=lfw.images)
print(f'  下载完成: {lfw.data.shape[0]} 样本, {lfw.data.shape[1]} 维特征')
" 2>/dev/null && info "  -> ${DATADIR}/lfw_people.npz" || warn "  LFW 下载失败"

    # ---- 6. Mall Customer Segmentation (Kaggle) ----
    step "6/9 Mall Customer Segmentation (Kaggle)"
    if command -v kaggle &> /dev/null; then
        kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python \
            -p "${DATADIR}" --unzip 2>/dev/null \
            && info "  -> ${DATADIR}/Mall_Customers.csv" \
            || warn "  Kaggle 下载失败"
    else
        warn "  未安装 Kaggle CLI"
        warn "  手动下载: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python"
    fi

    # ---- 7. Home Credit Default Risk (Kaggle) ----
    step "7/9 Home Credit Default Risk (Kaggle)"
    if command -v kaggle &> /dev/null; then
        kaggle competitions download -c home-credit-default-risk \
            -p "${DATADIR}" 2>/dev/null \
            && info "  -> ${DATADIR}/home-credit-default-risk.zip" \
            || warn "  Kaggle 下载失败，请先接受比赛规则: https://www.kaggle.com/competitions/home-credit-default-risk"
    else
        warn "  未安装 Kaggle CLI"
        warn "  手动下载: https://www.kaggle.com/competitions/home-credit-default-risk/data"
    fi

    # ---- 8. KDD Cup 1999 (直接下载) ----
    step "8/9 KDD Cup 1999 Dataset"
    KDD_URL="http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
    if command -v wget &> /dev/null; then
        wget -q "${KDD_URL}" -O "${DATADIR}/kddcup.data_10_percent.gz" 2>/dev/null \
            && info "  -> ${DATADIR}/kddcup.data_10_percent.gz" \
            || warn "  KDD Cup 下载失败"
    elif command -v curl &> /dev/null; then
        curl -sL "${KDD_URL}" -o "${DATADIR}/kddcup.data_10_percent.gz" 2>/dev/null \
            && info "  -> ${DATADIR}/kddcup.data_10_percent.gz" \
            || warn "  KDD Cup 下载失败"
    else
        warn "  需要 wget 或 curl"
        warn "  手动下载: ${KDD_URL}"
    fi

    # ---- 9. Adult Income Dataset (UCI) ----
    step "9/9 Adult Income Dataset (UCI)"
    ADULT_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    if command -v wget &> /dev/null; then
        wget -q "${ADULT_URL}" -O "${DATADIR}/adult.data" 2>/dev/null \
            && info "  -> ${DATADIR}/adult.data" \
            || warn "  Adult Income 下载失败"
    elif command -v curl &> /dev/null; then
        curl -sL "${ADULT_URL}" -o "${DATADIR}/adult.data" 2>/dev/null \
            && info "  -> ${DATADIR}/adult.data" \
            || warn "  Adult Income 下载失败"
    else
        warn "  需要 wget 或 curl"
        warn "  手动下载: ${ADULT_URL}"
    fi

    echo ""
    info "========== 数据集下载完成 =========="
    info "数据集目录: ${DATADIR}/"
    info "Kaggle 数据集需要配置 API Token:"
    info "  1. 登录 https://www.kaggle.com/ -> Account -> Create New Token"
    info "  2. 将 kaggle.json 放入 ~/.kaggle/ 目录"
    info "  3. chmod 600 ~/.kaggle/kaggle.json"
}

# ============================================================================
# 主入口
# ============================================================================
main() {
    local platform="${1:-help}"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  RAPIDS 云端环境配置脚本 v${RAPIDS_VERSION}${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    case "${platform}" in
        colab)      setup_colab ;;
        sagemaker)  setup_sagemaker ;;
        azure)      setup_azure ;;
        conda)      setup_conda ;;
        docker)     setup_docker ;;
        verify)     verify_rapids ;;
        datasets)   download_datasets "${2:-./datasets}" ;;
        all)
            setup_conda
            verify_rapids
            download_datasets
            ;;
        help|--help|-h|"")  usage ;;
        *)
            error "未知平台: ${platform}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
