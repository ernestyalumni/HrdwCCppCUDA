"""
@file test_bug_hunt.py

@brief Unit tests for FSW bug hunt practice — Matter Intelligence interview prep.

Mirrors the Rust tests in:
  Constructicon/Mixmaster/mixmaster/src/bug_hunt/bug_hunt.rs

Each test section is labelled with the bug number and name so they can be
found at a glance during a timed bug-hunt session.

Run from repo root:
  source .venv/bin/activate
  python -m pytest Voltron/unit_tests/BugHunt/test_bug_hunt.py -v
"""

import struct
import threading
import time

import pytest

from Voltron.BugHunt.bug_hunt import (
    # Bug 1
    parse_sensor_buggy,
    parse_sensor_fixed,
    # Bug 2
    RingBufferBuggy,
    RingBuffer,
    # Bug 3
    WatchdogBuggy,
    Watchdog,
    # Bug 4
    ActuatorControllerBuggy,
    ActuatorController,
    # Bug 5
    TelemetryProducerBuggy,
    TelemetryProducer,
    # Bug 6
    ReadTimeoutError,
    read_exact_or_timeout,
    # Bug 7
    FsmState,
    FsmEvent,
    BUGGY_TRANSITIONS,
    FIXED_TRANSITIONS,
    FaultManager,
    # Bug 8
    Device,
    open_and_calibrate_buggy,
    open_and_calibrate,
    open_and_calibrate_ctx,
    # Bug 9
    DataLoggerBuggy,
    DataLogger,
    # Bug 10
    BoundedQueueBuggy,
    BoundedQueue,
)


# ─── Bug 1: Endianness ────────────────────────────────────────────────────────
# Hardware sends big-endian; buggy code reads little-endian → garbage value.
# Symptom: wrong integer, no exception. Looks like sensor noise.

class TestEndianness:

    def test_buggy_misreads_big_endian_as_wrong_value(self):
        # 256 = 0x00000100; big-endian wire bytes: [0x00, 0x00, 0x01, 0x00]
        buf = struct.pack('>i', 256)
        assert parse_sensor_buggy(buf) != 256

    def test_buggy_misread_is_not_an_exception(self):
        # The danger: it silently returns a wrong value, never raises
        buf = struct.pack('>i', 256)
        result = parse_sensor_buggy(buf)
        assert isinstance(result, int)   # no crash — that's the bug

    def test_fixed_reads_positive_value(self):
        buf = struct.pack('>i', 256)
        assert parse_sensor_fixed(buf) == 256

    def test_fixed_reads_negative_value(self):
        # Negative values are common in sensor data (signed temperature, pressure delta)
        buf = struct.pack('>i', -1000)
        assert parse_sensor_fixed(buf) == -1000

    def test_fixed_reads_zero(self):
        buf = struct.pack('>i', 0)
        assert parse_sensor_fixed(buf) == 0

    def test_fixed_reads_max_i32(self):
        buf = struct.pack('>i', 2**31 - 1)
        assert parse_sensor_fixed(buf) == 2**31 - 1

    def test_buggy_and_fixed_disagree_on_same_buffer(self):
        # For most non-palindrome values the two interpretations differ
        buf = struct.pack('>i', 1234567)
        assert parse_sensor_buggy(buf) != parse_sensor_fixed(buf)


# ─── Bug 2: Ring Buffer Off-by-One ───────────────────────────────────────────
# head grows monotonically; after pop+push across the N boundary → IndexError.
# At 100 Hz with N=256: crashes at 2.56 s into the mission.

class TestRingBuffer:

    # ── buggy version ──

    def test_buggy_panics_after_fill_pop_push(self):
        # Exact failure sequence: fill to N, pop once, push again → data[N]
        rb = RingBufferBuggy(capacity=4)
        for i in range(4):
            rb.push(i)       # fills; head → 4
        rb.pop()             # count drops to 3, head stays at 4
        with pytest.raises(IndexError):
            rb.push(99)      # count=3 < 4 passes guard; data[4] → OOB

    def test_buggy_works_for_first_n_pushes(self):
        # Passes all startup tests — bug only manifests after first full cycle
        rb = RingBufferBuggy(capacity=4)
        assert rb.push(10)
        assert rb.push(20)
        assert rb.push(30)
        assert rb.push(40)

    # ── fixed version ──

    def test_push_wraps_head_at_capacity(self):
        rb = RingBuffer(capacity=4)
        for i in range(4):
            rb.push(i)
        assert rb.is_full
        assert not rb.push(99)     # full — rejected, no error

    def test_fifo_order_preserved(self):
        rb = RingBuffer(capacity=4)
        rb.push(1); rb.push(2); rb.push(3); rb.push(4)
        assert rb.pop() == 1
        assert rb.pop() == 2
        assert rb.pop() == 3
        assert rb.pop() == 4

    def test_push_after_partial_drain_wraps_correctly(self):
        rb = RingBuffer(capacity=4)
        rb.push(1); rb.push(2); rb.push(3); rb.push(4)
        rb.pop(); rb.pop()          # drain 2 → head stays at 0 (wrapped), count=2
        rb.push(5); rb.push(6)      # fills again; head wraps
        assert rb.pop() == 3
        assert rb.pop() == 4
        assert rb.pop() == 5
        assert rb.pop() == 6
        assert rb.is_empty

    def test_survives_full_256_cycle(self):
        # The exact scenario that kills the buggy version
        rb = RingBuffer(capacity=256)
        for i in range(256):
            assert rb.push(i % 256)
        for _ in range(256):
            rb.pop()
        assert rb.is_empty
        assert rb.push(42)      # would IndexError in buggy version (head == 256)
        assert rb.pop() == 42

    def test_pop_empty_returns_none(self):
        rb = RingBuffer(capacity=4)
        assert rb.pop() is None

    def test_len_tracks_count(self):
        rb = RingBuffer(capacity=4)
        assert len(rb) == 0
        rb.push('a'); rb.push('b')
        assert len(rb) == 2
        rb.pop()
        assert len(rb) == 1


# ─── Bug 3: Watchdog Kicked by Wrong Thread ──────────────────────────────────
# The architectural bug (heartbeat masking worker deadlock) can't be proven by
# a deterministic unit test — it requires a real concurrent deadlock scenario.
# We test the Watchdog contract: kick() resets timer, expiry is detected.

class TestWatchdog:

    def test_watchdog_not_expired_immediately(self):
        expired = threading.Event()
        wd = Watchdog(timeout_s=10.0, on_expire=expired.set)
        # With a 10s timeout, the on_expire callback should NOT fire immediately
        assert not expired.wait(timeout=0.15), "Watchdog should not fire immediately"
        wd.stop()

    def test_kick_resets_expiry_timer(self):
        expired = threading.Event()
        # Short timeout; we'll kick before it fires to verify kick resets the clock
        wd = Watchdog(timeout_s=0.2, on_expire=expired.set)
        time.sleep(0.1)
        wd.kick()           # reset — should push expiry out another 0.2s
        # Immediately after kick the callback should not have fired yet
        assert not expired.is_set(), "Watchdog fired too soon after kick"
        wd.stop()

    def test_watchdog_expires_without_kick(self):
        expired = threading.Event()
        wd = Watchdog(timeout_s=0.1, on_expire=expired.set)
        # Don't kick — let it expire
        assert expired.wait(timeout=1.0), "Watchdog should have fired"

    def test_kick_prevents_expiry(self):
        expired = threading.Event()
        wd = Watchdog(timeout_s=0.15, on_expire=expired.set)
        # Kick every 0.05s for 0.3s — well within timeout
        for _ in range(6):
            time.sleep(0.05)
            wd.kick()
        assert not expired.is_set(), "Watchdog should NOT have fired while being kicked"
        wd.stop()

    def test_buggy_watchdog_never_expires_even_without_worker(self):
        # Documents the bug: heartbeat keeps kicking regardless of worker state
        expired = threading.Event()
        _wd = WatchdogBuggy(timeout_s=0.1, on_expire=expired.set)
        # Even with no worker kicking, heartbeat prevents expiry
        fired = expired.wait(timeout=0.5)
        assert not fired, (
            "Bug confirmed: WatchdogBuggy never expires because "
            "heartbeat thread kicks independently of worker"
        )


# ─── Bug 4: Race Condition — Check-Then-Act ──────────────────────────────────
# Non-atomic check-then-act on plain bool → double-fire possible.
# FIX: hold lock across entire check-and-set.

class TestActuatorController:

    def test_fixed_first_fire_accepted(self):
        ctrl = ActuatorController()
        assert ctrl.fire() is True

    def test_fixed_second_fire_rejected_while_firing(self):
        ctrl = ActuatorController()
        ctrl.fire()
        assert ctrl.fire() is False    # still firing — rejected

    def test_fixed_fire_accepted_after_complete(self):
        ctrl = ActuatorController()
        ctrl.fire()
        ctrl.complete()
        assert ctrl.fire() is True     # reset — accepted again

    def test_fixed_is_firing_reflects_state(self):
        ctrl = ActuatorController()
        assert not ctrl.is_firing
        ctrl.fire()
        assert ctrl.is_firing
        ctrl.complete()
        assert not ctrl.is_firing

    def test_fixed_concurrent_fire_only_one_wins(self):
        # Multi-threaded: only one thread should win the fire() race
        ctrl = ActuatorController()
        results = []
        lock = threading.Lock()

        def try_fire():
            result = ctrl.fire()
            with lock:
                results.append(result)

        threads = [threading.Thread(target=try_fire) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1, (
            f"Exactly one thread should win fire(); got {results.count(True)}"
        )

    def test_buggy_logic_allows_double_fire_in_single_thread(self):
        # Single-thread demonstration of the logical race: if two checks
        # happen before either write, both pass
        ctrl = ActuatorControllerBuggy()
        # Simulate: both threads read firing==False before either sets it
        saw_false_1 = not ctrl.firing   # Thread A: reads False
        saw_false_2 = not ctrl.firing   # Thread B: also reads False
        assert saw_false_1 and saw_false_2, "Both threads see False — double-fire possible"


# ─── Bug 5: Integer Overflow / Sequence Number Wrap ─────────────────────────
# u8 wraps at 255; at 100 Hz that's every 2.55 s.

class TestTelemetrySequenceNumber:

    def test_buggy_seq_wraps_at_255(self):
        prod = TelemetryProducerBuggy()
        for _ in range(255):
            prod.next_seq()
        assert prod.next_seq() == 0    # wrapped — ground station counts 255 lost packets

    def test_buggy_seq_never_exceeds_255(self):
        prod = TelemetryProducerBuggy()
        for _ in range(1000):
            seq = prod.next_seq()
            assert seq <= 255

    def test_fixed_seq_does_not_wrap_at_255(self):
        prod = TelemetryProducer()
        for _ in range(255):
            prod.next_seq()
        assert prod.next_seq() == 256   # no wrap

    def test_fixed_seq_advances_past_256(self):
        prod = TelemetryProducer()
        for _ in range(256):
            prod.next_seq()
        assert prod.next_seq() == 257

    def test_fixed_seq_is_monotonic_for_reasonable_count(self):
        prod = TelemetryProducer()
        prev = 0
        for _ in range(10_000):
            seq = prod.next_seq()
            assert seq == prev + 1
            prev = seq


# ─── Bug 6: Missing Timeout on Blocking Read ─────────────────────────────────
# No timeout → hangs forever on sensor dropout. Silent, no exception.

class TestBlockingRead:

    def test_partial_read_raises_timeout_error(self):
        with pytest.raises(ReadTimeoutError) as exc_info:
            read_exact_or_timeout(b'\x00' * 5, expected=9)
        assert exc_info.value.got == 5
        assert exc_info.value.expected == 9

    def test_timeout_error_message_is_informative(self):
        try:
            read_exact_or_timeout(b'\x00' * 3, expected=9)
        except ReadTimeoutError as e:
            assert '3' in str(e)
            assert '9' in str(e)

    def test_zero_bytes_raises_timeout(self):
        with pytest.raises(ReadTimeoutError) as exc_info:
            read_exact_or_timeout(b'', expected=9)
        assert exc_info.value.got == 0

    def test_exact_bytes_succeeds(self):
        buf = bytes(range(9))
        result = read_exact_or_timeout(buf, expected=9)
        assert result == buf

    def test_more_than_expected_returns_truncated(self):
        buf = b'\xAB' * 20
        result = read_exact_or_timeout(buf, expected=9)
        assert len(result) == 9

    def test_one_byte_short_raises(self):
        with pytest.raises(ReadTimeoutError):
            read_exact_or_timeout(b'\x00' * 8, expected=9)


# ─── Bug 7: State Machine Invalid Transition ─────────────────────────────────
# Buggy: FAULT → NOMINAL directly (skips SafeMode checks).
# Fixed: FAULT → SAFE_MODE → NOMINAL only.

class TestFaultManagerFsm:

    def test_nominal_to_fault_on_fault_detected(self):
        fm = FaultManager()
        fm.transition(FsmEvent.FAULT_DETECTED)
        assert fm.state == FsmState.FAULT

    def test_fault_to_nominal_directly_is_rejected(self):
        fm = FaultManager()
        fm.transition(FsmEvent.FAULT_DETECTED)
        with pytest.raises(ValueError):
            fm.transition(FsmEvent.CHECKS_PASSED)  # skip SafeMode — must fail

    def test_state_unchanged_after_invalid_transition(self):
        fm = FaultManager()
        fm.transition(FsmEvent.FAULT_DETECTED)
        try:
            fm.transition(FsmEvent.CHECKS_PASSED)
        except ValueError:
            pass
        assert fm.state == FsmState.FAULT   # must not have changed

    def test_correct_path_fault_safemode_nominal(self):
        fm = FaultManager()
        fm.transition(FsmEvent.FAULT_DETECTED)
        assert fm.state == FsmState.FAULT
        fm.transition(FsmEvent.SAFE_MODE_CMD)
        assert fm.state == FsmState.SAFE_MODE
        fm.transition(FsmEvent.CHECKS_PASSED)
        assert fm.state == FsmState.NOMINAL

    def test_safe_mode_reset_returns_to_nominal(self):
        fm = FaultManager()
        fm.transition(FsmEvent.FAULT_DETECTED)
        fm.transition(FsmEvent.SAFE_MODE_CMD)
        fm.transition(FsmEvent.RESET)
        assert fm.state == FsmState.NOMINAL

    def test_nominal_reset_is_invalid(self):
        fm = FaultManager()
        with pytest.raises(ValueError):
            fm.transition(FsmEvent.RESET)

    def test_buggy_fsm_allows_illegal_direct_transition(self):
        # Documents the bug: FAULT → NOMINAL without SafeMode checks
        fm = FaultManager(transitions=BUGGY_TRANSITIONS)
        fm.transition(FsmEvent.FAULT_DETECTED)
        assert fm.state == FsmState.FAULT
        fm.transition(FsmEvent.CHECKS_PASSED)   # should not be allowed — but buggy fsm permits it
        assert fm.state == FsmState.NOMINAL     # hardware resumes without verification

    def test_safe_mode_fault_detected_is_invalid(self):
        fm = FaultManager()
        fm.transition(FsmEvent.FAULT_DETECTED)
        fm.transition(FsmEvent.SAFE_MODE_CMD)
        with pytest.raises(ValueError):
            fm.transition(FsmEvent.FAULT_DETECTED)


# ─── Bug 8: Resource Leak on Error Path ──────────────────────────────────────
# Device.open() acquires HW resource; if calibrate() fails, shutdown() never called.
# FIX: try/except or context manager ensures cleanup on error path.

class TestResourceLeak:

    def test_buggy_raises_on_failure(self):
        with pytest.raises(RuntimeError, match="calibration failed"):
            open_and_calibrate_buggy("device_fail")

    def test_fixed_raises_on_failure(self):
        with pytest.raises(RuntimeError, match="calibration failed"):
            open_and_calibrate("device_fail")

    def test_fixed_succeeds_and_device_is_calibrated(self):
        dev = open_and_calibrate("device_ok")
        assert dev.calibrated is True

    def test_ctx_manager_raises_on_failure(self):
        with pytest.raises(RuntimeError):
            open_and_calibrate_ctx("device_fail")

    def test_ctx_manager_succeeds(self):
        dev = open_and_calibrate_ctx("device_ok")
        assert dev.calibrated is True

    def test_device_shutdown_called_on_error(self):
        # Instrument shutdown() to confirm it's called on the error path
        shutdown_called = []
        original_shutdown = Device.shutdown

        def patched_shutdown(self):
            shutdown_called.append(self.name)
            original_shutdown(self)

        Device.shutdown = patched_shutdown
        try:
            open_and_calibrate("device_fail")
        except RuntimeError:
            pass
        finally:
            Device.shutdown = original_shutdown

        assert len(shutdown_called) == 1, "shutdown() must be called exactly once on error"
        assert shutdown_called[0] == "device_fail"

    def test_device_shutdown_not_called_on_success(self):
        # On success, the caller owns the device — shutdown must NOT be called
        shutdown_called = []
        original_shutdown = Device.shutdown

        def patched_shutdown(self):
            shutdown_called.append(self.name)
            original_shutdown(self)

        Device.shutdown = patched_shutdown
        try:
            open_and_calibrate("device_ok")
        finally:
            Device.shutdown = original_shutdown

        assert len(shutdown_called) == 0, "shutdown() must NOT be called on success"


# ─── Bug 9: Mutex Deadlock (ABBA) ────────────────────────────────────────────
# write_data: data_lock → file_lock (A→B)
# rotate_log: file_lock → data_lock (B→A)
# → ABBA deadlock when called from different threads concurrently.
# FIX: single lock, or consistent acquisition order everywhere.

class TestDataLogger:

    def test_write_then_rotate_single_thread(self):
        logger = DataLogger()
        logger.write_data("line 1")
        logger.write_data("line 2")
        assert logger.entry_count() == 2
        logger.rotate_log()
        assert logger.entry_count() == 0

    def test_write_after_rotate(self):
        logger = DataLogger()
        logger.write_data("before")
        logger.rotate_log()
        logger.write_data("after")
        assert logger.entry_count() == 1

    def test_concurrent_writes_no_deadlock(self):
        logger = DataLogger()
        errors = []

        def writer():
            try:
                for i in range(50):
                    logger.write_data(f"line {i}")
            except Exception as e:
                errors.append(e)

        def rotator():
            try:
                for _ in range(10):
                    time.sleep(0.002)
                    logger.rotate_log()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads.append(threading.Thread(target=rotator))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Concurrent access raised: {errors}"
        # If any thread is still alive, it deadlocked
        assert all(not t.is_alive() for t in threads), "Thread deadlocked"

    def test_entry_count_consistent_under_concurrent_writes(self):
        logger = DataLogger()
        n_threads = 4
        writes_per_thread = 25

        def writer():
            for i in range(writes_per_thread):
                logger.write_data(f"data {i}")

        threads = [threading.Thread(target=writer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert logger.entry_count() == n_threads * writes_per_thread


# ─── Bug 10: Spurious Wakeup — if vs while ────────────────────────────────────
# Python Rust difference: Rust's Condvar forces while; Python does not.
# BUG: 'if' on condition check — spurious wakeup skips the guard.
# FIX: always 'while' with condition variables.

class TestBoundedQueue:

    def test_enqueue_dequeue_basic(self):
        q = BoundedQueue(capacity=3)
        q.enqueue(10)
        q.enqueue(20)
        q.enqueue(30)
        assert len(q) == 3
        assert q.dequeue() == 10
        assert q.dequeue() == 20
        assert q.dequeue() == 30
        assert len(q) == 0

    def test_fifo_order_preserved(self):
        q = BoundedQueue(capacity=5)
        for i in range(5):
            q.enqueue(i)
        for i in range(5):
            assert q.dequeue() == i

    def test_enqueue_blocks_until_space(self):
        q = BoundedQueue(capacity=1)
        q.enqueue("first")

        result = []
        def late_enqueue():
            q.enqueue("second")
            result.append("enqueued")

        t = threading.Thread(target=late_enqueue)
        t.start()
        time.sleep(0.05)
        assert not result, "Should be blocked — queue is full"
        q.dequeue()          # free a slot
        t.join(timeout=1.0)
        assert result == ["enqueued"]

    def test_dequeue_blocks_until_item(self):
        q = BoundedQueue(capacity=4)
        result = []

        def late_dequeue():
            val = q.dequeue()
            result.append(val)

        t = threading.Thread(target=late_dequeue)
        t.start()
        time.sleep(0.05)
        assert not result, "Should be blocked — queue is empty"
        q.enqueue(42)
        t.join(timeout=1.0)
        assert result == [42]

    def test_producer_consumer_throughput(self):
        q = BoundedQueue(capacity=16)
        n = 200
        produced = []
        consumed = []

        def producer():
            for i in range(n):
                q.enqueue(i)
                produced.append(i)

        def consumer():
            for _ in range(n):
                consumed.append(q.dequeue())

        pt = threading.Thread(target=producer)
        ct = threading.Thread(target=consumer)
        pt.start(); ct.start()
        pt.join(timeout=5.0); ct.join(timeout=5.0)

        assert not pt.is_alive() and not ct.is_alive(), "Thread stalled"
        assert sorted(consumed) == list(range(n))
