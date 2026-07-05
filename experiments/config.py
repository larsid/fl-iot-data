SERVER_IMAGE = "fl-server:latest"
CLIENT_IMAGE = "fl-client:latest"

FL_PARAMS = {
    "algorithm": "FedAvg",
    "num_clients": 8,
    "num_rounds": 500,
    "local_epochs": 2,
    "batch_size": 16,
    "learning_rate": 0.01,
    "optimizer": "SGD",
    "dataset": "CIFAR-10",
    "partition": "IID",
    "model": "CNN3",
}

WORKER_IP = "127.0.0.1"
WORKER_PORT = 5000
FL_SERVER_PORT = 9191
NUM_CLIENTS = 8
REPETITIONS = 5

SERVER_RESOURCES = {
    "cpus": 2,
    "mem_limit": "4096m",
    "memswap_limit": "4096m",
    "cpuset_cpus": "0-3",
}
SERVER_BANDWIDTH = {"bw": 1000}

CLIENT_BASELINE = {}
BANDWIDTH_BASELINE = {}

CLIENT_LEVEL1 = {
    "cpus": 1,
}
BANDWIDTH_LEVEL1 = {}

CLIENT_LEVEL2 = {
    "cpus": 1,
    "mem_limit": "1024m",
    "memswap_limit": "1024m",
}
BANDWIDTH_LEVEL2 = {}

CLIENT_LEVEL3 = {
    "cpus": 1,
    "mem_limit": "1024m",
    "memswap_limit": "1024m",
}
BANDWIDTH_LEVEL3 = {"bw": 100}
