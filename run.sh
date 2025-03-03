#!/usr/bin/env bash
pip install -r requirements.txt
mkdir -p transcripts
python qlearning.py
python audit.py
python draw.py
python audit_t.py
python draw_t.py
