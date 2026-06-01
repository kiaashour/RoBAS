import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _str2bool(value: str) -> bool:
    """Parse a CLI boolean argument.

    Parameters
    ----------
    value : str
        The input value to parse.

    Returns
    -------
    bool
        The parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_n_cal(value: str):
    """Parse calibration size as int or the literal 'standard'.

    Parameters
    ----------
    value : str
        The input value to parse.

    Returns
    -------
    int
        The parsed calibration size.
    """
    if isinstance(value, int):
        return value
    if value == "standard":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("n_cal must be an integer or 'standard'.") from exc    