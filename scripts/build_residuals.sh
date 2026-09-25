#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/build"
for model in particle pusher tipover_push hopper; do
    "${CXX:-g++}" -O3 -std=c++17 -fPIC -shared \
        "$repo_root/cpp/${model}_residual.cpp" \
        -o "$repo_root/build/lib${model}_residual.so"
done
