from Voltron.DesignProblems.RateLimiter.TokenBucketRateLimiters import (
    PerUserTokenBucket,
    TokenBucket,
)

class FakeTime:
    def __init__(self, now = 0.0):
        self.now = now
    
    def time(self):
        return self.now

    def advance(self, time_elapsed):
        self.now += time_elapsed

def test_token_bucket_rate_limiter():
    fake_time = FakeTime()

    # 1 token/sec
    bucket = TokenBucket(fake_time, capacity=3, refill_rate=1.0)
    t = fake_time.now

    # Starts full: 3 tokens.
    assert bucket.allow(now=t) is True   # 2 left
    assert bucket.allow(now=t) is True   # 1 left
    assert bucket.allow(now=t) is True   # 0 left
    assert bucket.allow(now=t) is False  # no tokens

    # After 0.5s, <1 token: still no.
    fake_time.advance(0.5)
    assert bucket.allow(now=fake_time.time()) is False

    # After another 0.5s (total 1s): ~1 token refilled.
    fake_time.advance(0.5)
    assert bucket.allow(now=fake_time.time()) is True   # consume the 1 token
    assert bucket.allow(now=fake_time.time()) is False  # empty again

def test_token_bucket_does_not_exceed_capacity():
    fake_time = FakeTime()
    bucket = TokenBucket(fake_time, capacity=2, refill_rate=10.0)
    t = fake_time.time()

    # Use both tokens.
    assert bucket.allow(now=t) is True
    assert bucket.allow(now=t) is True
    assert bucket.allow(now=t) is False

    # Wait "a long time": tokens should cap at capacity=2, not grow unbounded.
    fake_time.advance(100.0)
    assert bucket.allow(now=fake_time.time()) is True
    assert bucket.allow(now=fake_time.time()) is True
    assert bucket.allow(now=fake_time.time()) is False

def test_per_user_independent_buckets():
    fake_time = FakeTime()
    limiter = PerUserTokenBucket(fake_time, capacity=2, refill_rate=1.0)
    t = fake_time.time()

    # userA uses up their 2 tokens
    assert limiter.allow("userA", now=t) is True
    assert limiter.allow("userA", now=t) is True
    assert limiter.allow("userA", now=t) is False

    # userB has a fresh bucket
    assert limiter.allow("userB", now=t) is True
    assert limiter.allow("userB", now=t) is True
    assert limiter.allow("userB", now=t) is False

def test_per_user_refill():
    fake_time = FakeTime()
    limiter = PerUserTokenBucket(fake_time, capacity=1, refill_rate=1.0)
    t = fake_time.time()

    # consume only token
    assert limiter.allow("userA", now=t) is True
    # no tokens
    assert limiter.allow("userA", now=t) is False

    # >1s later, should refill ~1 token
    fake_time.advance(1.1)
    # allowed again
    assert limiter.allow("userA", now=fake_time.time()) is True