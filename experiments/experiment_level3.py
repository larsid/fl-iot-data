#!/usr/bin/env python3
from config import BANDWIDTH_LEVEL3, CLIENT_LEVEL3
from experiment_runner import run_experiment

run_experiment(
    level_label="3",
    client_resources=CLIENT_LEVEL3,
    client_bandwidth=BANDWIDTH_LEVEL3,
    controlled=True,
)
