## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I A2(8*64G)推理设备：支持的卡数最小为1
- [Atlas 800I A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 1.3 环境依赖安装
```shell
pip3 install -r requirements.txt
```

### 1.4 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.5 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

## 二、下载权重

### 2.1 权重及配置文件说明
1. Wan2.1-T2V-1.3B权重链接:
```shell
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
```
2. Wan2.1-T2V-14B权重链接
```shell
https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
```
3. Wan2.1-I2V-480P权重链接:
```shell
https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
```
4. Wan2.1-I2V-720P权重链接
```shell
https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
```

## 三、Wan2.1使用

### 3.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/Wan2.1.git
cd Wan2.1
```

### 3.2 Wan2.1-T2V-1.3B

使用上一步下载的权重
```shell
model_base="./Wan2.1-T2V-1.3B/"
```
#### 3.2.1 单卡性能测试
执行命令：
```shell
# Wan2.1-T2V-1.3B
python generate.py  \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ${model_base} \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```
参数说明：
- task: 任务类型。
- ckpt_dir: 模型的权重路径
- size: 生成视频的高和宽
- prompt: 文本提示词

#### 3.2.2 多卡性能测试
执行命令：
```shell
# 1.3B支持单卡、双卡、四卡
torchrun --nproc_per_node=4 generate.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir ${model_base} \
--dit_fsdp \
--t5_fsdp \
--ulysses_size 4 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```
参数说明：
- dit_fsdp: DiT使用FSDP
- t5_fsdp: T5使用FSDP
- ulysses_size: ulysses并行数

### 3.3 Wan2.1-T2V-14B
使用上一步下载的权重
```shell
model_base="./Wan2.1-T2V-14B/"
```

#### 3.3.1 8卡性能测试
执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=8 generate.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir ${model_base} \
--dit_fsdp \
--t5_fsdp \
--ulysses_size 8 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- ulysses_size: ulysses并行数

#### 3.3.2 16卡性能测试
执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=16 generate.py \
--task t2v-14B \
--size 1280*720 \
--ckpt_dir ${model_base} \
--dit_fsdp \
--t5_fsdp \
--ring_size 2 \
--ulysses_size 8 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- ring_size: ring并行数

### 3.4 Wan2.1-I2V-14B
使用上一步下载的权重
```shell
# 生成480P的视频
model_base="./Wan2.1-I2V-14B-480P/"
# 生成720P的视频
model_base="./Wan2.1-I2V-14B-720P/"
```

#### 3.3.1 8卡性能测试

执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=8 generate.py \
--task i2v-14B \
--size 832*480 \
--ckpt_dir ${model_base} \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--ulysses_size 8 \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- task: 任务类型。
- size: 生成视频的分辨率，支持[1280, 720]、[832, 480]、[720, 480]等
- ckpt_dir: 模型的权重路径
- frame_num: 生成视频的帧数
- sample_steps: 推理步数
- dit_fsdp: DiT使用FSDP
- t5_fsdp: T5使用FSDP
- ulysses_size: ulysses并行数
- image: 用于生成视频的图片路径
- prompt: 文本提示词

#### 3.3.2 16卡性能测试
执行命令：
```shell
export ALGO=0
torchrun --nproc_per_node=16 generate.py \
--task i2v-14B \
--size 832*480 \
--ckpt_dir ${model_base} \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--ring_size 2 \
--ulysses_size 8 \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- ring_size: ring并行数

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。