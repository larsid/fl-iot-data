FROM python:3.9-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        iproute2 \
        iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==1.13.1 torchvision==0.14.1 \
 && pip install \
        'flwr[simulation]==1.11.0' \
        numpy==1.24.4

COPY data/cifar-10-batches-py /data/cifar-10-batches-py

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app DATA_ROOT=/data
ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "-c", "import torch, flwr, ray, os; print('torch', torch.__version__); print('flwr', flwr.__version__); print('ray', ray.__version__); print('cifar files:', os.listdir('/data/cifar-10-batches-py')[:3])"]
