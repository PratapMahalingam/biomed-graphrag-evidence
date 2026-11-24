#!/usr/bin/env bash
# SciFact (Wadden et al., EMNLP 2020), CC BY-NC 2.0. Writes data/scifact/{corpus,claims_*}.jsonl
set -euo pipefail
mkdir -p data
curl -sSL -o data/scifact.tar.gz https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
tar -xzf data/scifact.tar.gz -C data
rm -rf data/scifact && mv data/data data/scifact && rm data/scifact.tar.gz
wc -l data/scifact/*.jsonl
