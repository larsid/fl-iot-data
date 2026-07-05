#!/usr/bin/env python3
from config import BANDWIDTH_BASELINE, CLIENT_BASELINE
from experiment_runner import run_experiment

run_experiment(
    level_label="0",
    client_resources=CLIENT_BASELINE,
    client_bandwidth=BANDWIDTH_BASELINE,
    controlled=False,
)
