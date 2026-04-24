import os



def int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default

def float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default

def bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}
