#!/usr/bin/env python3
from config import BANDWIDTH_LEVEL2, CLIENT_LEVEL2
from experiment_runner import run_experiment

run_experiment(
    level_label="2",
    client_resources=CLIENT_LEVEL2,
    client_bandwidth=BANDWIDTH_LEVEL2,
    controlled=True,
)
