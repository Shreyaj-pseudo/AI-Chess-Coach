"""Request serialization and rate-limit state shared by the Lichess adapters."""

from email.utils import parsedate_to_datetime
import math
import threading
import time


REQUEST_LOCK = threading.Lock()
COOLDOWN_UNTIL = 0.0


def retry_seconds(header):
    try:
        seconds = float(header)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(header).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            seconds = 60
    return max(60, math.ceil(seconds)) if math.isfinite(seconds) else 60
