from collections import OrderedDict
from dataclasses import dataclass
import math
from threading import Lock
import time
from typing import Callable


@dataclass(frozen=True)
class RateLimitPolicy:
    bucket: str
    limit: int
    window_seconds: int = 60

    def __post_init__(self) -> None:
        if self.limit < 1 or self.window_seconds < 1:
            raise ValueError("Rate limits and windows must be positive integers.")


class RateLimitExceeded(Exception):
    def __init__(
        self, bucket: str, limit: int, retry_after: int, reset_at: int
    ) -> None:
        self.bucket = bucket
        self.limit = limit
        self.retry_after = retry_after
        self.reset_at = reset_at
        super().__init__(bucket)


class RateLimitStorageFull(Exception):
    pass


class IpRateLimiter:
    def __init__(
        self,
        max_entries: int = 50_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive.")
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], tuple[int, int]] = OrderedDict()
        self._lock = Lock()
        self._next_cleanup = 0.0

    def enforce(
        self, client_ip: str, policies: tuple[RateLimitPolicy, ...]
    ) -> None:
        now = self._clock()
        with self._lock:
            if now >= self._next_cleanup or len(self._entries) >= self._max_entries:
                expired = [
                    key
                    for key, (reset_at, _) in self._entries.items()
                    if reset_at <= now
                ]
                for key in expired:
                    del self._entries[key]
                self._next_cleanup = now + 60

            checks: list[tuple[tuple[str, str], RateLimitPolicy, int, int]] = []
            for policy in policies:
                key = (client_ip, policy.bucket)
                window_start = (
                    int(now // policy.window_seconds) * policy.window_seconds
                )
                reset_at = window_start + policy.window_seconds
                entry = self._entries.get(key)
                count = entry[1] if entry and entry[0] == reset_at else 0
                if count >= policy.limit:
                    raise RateLimitExceeded(
                        policy.bucket,
                        policy.limit,
                        max(1, math.ceil(reset_at - now)),
                        reset_at,
                    )
                checks.append((key, policy, reset_at, count))

            new_keys = sum(key not in self._entries for key, *_ in checks)
            if len(self._entries) + new_keys > self._max_entries:
                raise RateLimitStorageFull

            for key, _, reset_at, count in checks:
                self._entries[key] = (reset_at, count + 1)
                self._entries.move_to_end(key)
