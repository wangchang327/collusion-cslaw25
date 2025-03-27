#!/usr/bin/env bash
pip install -r requirements.txt
mkdir -p transcripts
python qlearning_t.py
python audit_t.py
python draw_t.py
