class TokenBucket:
    def __init__(self, time_module, capacity: int, refill_rate: float):
        """
        Start with allow for some number of tokens, defined by capacity.
        At any time, track the number of tokens available to use with "tokens."
        Refill rate increases the number of tokens available to use per time
        interval elapsed. Refilling allows for total number of tokens that can
        be used to increase, and it's ok to increase beyond the original
        capacity value.

        time_module: This could be an instance of time, from import time or a
        "fake" time module that at least has this function as an interface:
        time_module.time() -> float
        which provides the "current time".
        capacity: max tokens in the bucket (burst size)
        refill_rate: tokens added per second
        """
        self._time_module = time_module
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill_ts = self._time_module.time()

    def _refill(self, now: float) -> None:
        elapsed = now - self.last_refill_ts
        if elapsed <= 0:
            return
        added = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + added)
        self.last_refill_ts = now

    def allow(self, now: float | None = None) -> bool:
        """
        Return True if a request is allowed (consume one token),
        False if rate‑limited.
        """
        if now is None:
            now = self._time_module.time()
        self._refill(now)
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

class PerUserTokenBucket:
    def __init__(self, time_module, capacity: int, refill_rate: float):
        self._time_module = time_module
        # default capacity for each user.
        self.capacity = capacity
        # default refill rate for each user.
        self.refill_rate = refill_rate
        # map of user's unique name/identifier, to the user's token bucket.
        self._user_to_token_bucket: dict[str, TokenBucket] = {}

    def allow(self, user: str, now: float | None = None) -> bool:
        if user not in self._user_to_token_bucket:
            self._user_to_token_bucket[user] = TokenBucket(
                self._time_module,
                self.capacity,
                self.refill_rate)
        return self._user_to_token_bucket[user].allow(now)
 