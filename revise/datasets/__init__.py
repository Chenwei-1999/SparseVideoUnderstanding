"""Dataset and task-adapter boundary for REVISE evaluations.

Dataset modules own benchmark sample loading and dataset-facing adapters. Model
runtime choices such as vLLM HTTP or HuggingFace in-process live under
``revise.backends``.
"""
