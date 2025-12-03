#!/bin/sh
# Start backend
uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
