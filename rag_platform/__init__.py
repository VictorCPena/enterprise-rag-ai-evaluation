"""Local evidence application; disable optional SDK telemetry by default."""

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "true")
