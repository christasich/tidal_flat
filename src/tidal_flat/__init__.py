__title__ = "tidal_flat"
__author__ = "christasich"
__license__ = "MIT"
__version__ = "0.1.0"

from loguru import logger

from . import constants, data
from .core import inundation, model, platform, tides
from .core.model import Model
from .core.platform import Platform
from .core.tides import Tides

__all__ = ["constants", "tides", "data", "model", "platform", "inundation", "Tides", "Model", "Platform"]

logger.disable(__name__)
