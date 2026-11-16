"""edgevision - a small, modular object-detection pipeline for edge devices.

Stages (each one is its own module, so it can be tested and timed alone):

    capture  ->  preprocess  ->  inference (backend)  ->  postprocess  ->  visualize
"""

__version__ = "1.0.0"
