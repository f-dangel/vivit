import logging

LOGGER = None
GLOBAL_STEP = 0


def get_logger():
    """Get the global ``LOGGER``. If ``LOGGER`` is ``None``, initialize the logger
    by calling the ``init_logger`` function"""

    global LOGGER
    if LOGGER is None:
        LOGGER = init_logger()
    return LOGGER


def init_logger():
    """Initialize the a logger``."""

    # Create logger
    logger = logging.getLogger("lowrank")
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler("lowrank.log", mode="w")
    fh.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")

    # Add formatter to file handler
    fh.setFormatter(formatter)

    # Add file handler to logger
    logger.addHandler(fh)
    return logger


def set_global_step(new_step):
    """Set the global step to the given ``new_step``"""
    global GLOBAL_STEP
    GLOBAL_STEP = new_step


def get_global_step():
    """Return the global step"""
    return GLOBAL_STEP


def log_with_global_step(msg):
    """Write a log message including the global step."""

    msg_with_global_step = f"global_step={get_global_step()}:{msg}"
    logger = get_logger()
    logger.info(msg_with_global_step)


def tensor_to_list(tensor):
    """Convert a torch.Tensor to a list"""
    return tensor.detach().cpu().numpy().tolist()
