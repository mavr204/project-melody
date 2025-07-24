import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("numba").setLevel(logging.WARNING)
def get_logger(name: str):
    return logging.getLogger(name=name)