#!/usr/bin/env python3
from config import BANDWIDTH_LEVEL1, CLIENT_LEVEL1
from experiment_runner import run_experiment

run_experiment(
    level_label="1",
    client_resources=CLIENT_LEVEL1,
    client_bandwidth=BANDWIDTH_LEVEL1,
    controlled=True,
)
