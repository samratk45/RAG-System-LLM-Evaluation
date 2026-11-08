from .custom_logger import CustomLogger as _CustomLogger  
try:
    from .custom_logger import CustomLogger
except Exception:
    CustomLogger = _CustomLogger

GLOBAL_LOGGER = CustomLogger().get_logger(__name__)