"""Functions for setting up a logger, setting the global logging step ``GLOBAL_STEP``
and writing a ``torch.Tensor`` to the log-file.
"""

import logging

LOGGER = None
GLOBAL_STEP = 0


def get_logger():
    """Get the global ``LOGGER``. If ``LOGGER`` is ``None``, initialize the logger
    by calling the ``init_logger`` function.

    Returns:
        logging.Logger: The ``logging.Logger`` object created by init_logger and stored
            in the global variable ``LOGGER``
    """
    global LOGGER
    if LOGGER is None:
        LOGGER = init_logger()
    return LOGGER


def init_logger():
    """Initialize the logger.

    Returns:
        logging.Logger: The logger with the configuration specified below
    """

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
    """Set the global step to the given ``new_step``

    Args:
        new_step (int): Set the global logging step ``GLOBAL_STEP`` to ``new_step``
    """
    global GLOBAL_STEP
    GLOBAL_STEP = new_step


def get_global_step():
    """Return the global step

    Returns:
        int: The current global logging step ``GLOBAL_STEP``
    """
    return GLOBAL_STEP


def log_with_global_step(msg):
    """Write a log message including the global logging step.

    Args:
        msg (str): The logging message
    """
    msg_with_global_step = f"global_step={get_global_step()}:{msg}"
    logger = get_logger()
    logger.info(msg_with_global_step)


def tensor_to_list(T):
    """Convert a torch.Tensor to a list

    Args:
        T (torch.Tensor): The ``torch.Tensor`` that is supposed to be converted

    Returns:
        list: The tensor converted to a list
    """
    return T.detach().cpu().numpy().tolist()
