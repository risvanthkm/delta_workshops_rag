import unittest

from rate_limit import (
    IpRateLimiter,
    RateLimitExceeded,
    RateLimitPolicy,
    RateLimitStorageFull,
)


class IpRateLimiterTests(unittest.TestCase):
    def test_limit_is_per_ip_and_per_bucket(self) -> None:
        limiter = IpRateLimiter(clock=lambda: 10)
        embedding = RateLimitPolicy("embedding", limit=1)
        llm = RateLimitPolicy("llm", limit=1)

        limiter.enforce("192.0.2.1", (embedding,))
        limiter.enforce("192.0.2.2", (embedding,))
        limiter.enforce("192.0.2.1", (llm,))

        with self.assertRaises(RateLimitExceeded) as raised:
            limiter.enforce("192.0.2.1", (embedding,))
        self.assertEqual(raised.exception.retry_after, 50)

    def test_combined_limits_are_consumed_atomically(self) -> None:
        limiter = IpRateLimiter(clock=lambda: 10)
        embedding = RateLimitPolicy("embedding", limit=3)
        llm = RateLimitPolicy("llm", limit=1)

        limiter.enforce("192.0.2.1", (embedding,))
        limiter.enforce("192.0.2.1", (embedding, llm))
        with self.assertRaises(RateLimitExceeded):
            limiter.enforce("192.0.2.1", (embedding, llm))

        limiter.enforce("192.0.2.1", (embedding,))
        with self.assertRaises(RateLimitExceeded):
            limiter.enforce("192.0.2.1", (embedding,))

    def test_window_resets_and_storage_fails_closed_at_capacity(self) -> None:
        now = [59.9]
        limiter = IpRateLimiter(max_entries=1, clock=lambda: now[0])
        embedding = RateLimitPolicy("embedding", limit=1)

        limiter.enforce("192.0.2.1", (embedding,))
        now[0] = 60
        limiter.enforce("192.0.2.1", (embedding,))
        with self.assertRaises(RateLimitStorageFull):
            limiter.enforce("192.0.2.2", (embedding,))


if __name__ == "__main__":
    unittest.main()
