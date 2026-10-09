# Image for the DeepSeek-V4.1-Flash 2.86 bpw recipe: NVIDIA's PyTorch container plus this repository, installed editable.
# Build from the repository root:  docker build -f recipes/dsv41-2spark/Dockerfile -t tensorfold-dsv41:2.86 .
FROM nvcr.io/nvidia/pytorch:26.07-py3
COPY . /opt/tensorfold
RUN pip install --no-cache-dir -e /opt/tensorfold
