__title__ = "tidal_flat"
__author__ = "christasich"
__license__ = "MIT"
__version__ = "0.1.0"

from loguru import logger

from . import constants, data
from .core import cycle, inundation, model, platform, tides

__all__ = ["constants", "tides", "data", "model", "platform", "cycle", "inundation"]

logger.disable(__name__)
