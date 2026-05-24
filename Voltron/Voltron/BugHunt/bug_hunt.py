"""
Bug Hunt Practice — FSW Context (Matter Intelligence interview prep)
Python mirror of bug_hunt.rs

For each bug: read the BUGGY version, cover the FIX, find it yourself first.
Target: < 5 min per bug.
Verbalize: what you see → runtime symptom → fix → FSW impact.

Run tests: python -m pytest bug_hunt.py -v
"""

import struct
import threading
import time
from enum import Enum, auto
from typing import Optional


# ─── Bug 1: Endianness (Hardware → Software) ──────────────────────────────────
# Sensor sends 4-byte big-endian i32 over UART. ICD says: big-endian.
# Symptom: garbage values even when sensor is healthy — looks like sensor noise,
#   it's a software bug. Only discovered when you compare raw bytes to expected.

def parse_sensor_buggy(buf: bytes) -> int:
    # BUG: '<i' = little-endian, but hardware ICD specifies big-endian
    return struct.unpack('<i', buf[:4])[0]

def parse_sensor_fixed(buf: bytes) -> int:
    # FIX: '>i' = big-endian, matches hardware ICD
    # Rule: ALWAYS check the ICD at every HW/SW boundary. Never assume endianness.
    return struct.unpack('>i', buf[:4])[0]


# ─── Bug 2: Off-by-one in Ring Buffer ─────────────────────────────────────────
# head grows without wrapping → IndexError after capacity pushes.
# Symptom: works perfectly for first N pushes, then crashes.
# At 100 Hz with N=256: crashes at exactly 2.56 seconds into the mission.

class RingBufferBuggy:
    def __init__(self, capacity: int = 256):
        self._data = [None] * capacity
        self._capacity = capacity
        self._head = 0
        self._count = 0

    def push(self, val) -> bool:
        if self._count >= self._capacity:
            return False
        self._data[self._head] = val
        self._head += 1  # BUG: no modulo — head walks off the end of the array
        self._count += 1
        return True

    def pop(self) -> Optional[object]:
        if self._count == 0:
            return None
        tail = (self._head - self._count) % self._capacity
        self._count -= 1
        return self._data[tail]


class RingBuffer:
    def __init__(self, capacity: int = 256):
        self._data = [None] * capacity
        self._capacity = capacity
        self._head = 0   # next write position
        self._count = 0

    def push(self, val) -> bool:
        if self._count >= self._capacity:
            return False
        self._data[self._head] = val
        self._head = (self._head + 1) % self._capacity  # FIX: wrap around
        self._count += 1
        return True

    def pop(self) -> Optional[object]:
        # tail is count slots behind head.
        # Use (head + capacity - count) % capacity to avoid negative usize equivalent.
        if self._count == 0:
            return None
        tail = (self._head + self._capacity - self._count) % self._capacity
        self._count -= 1
        return self._data[tail]

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    @property
    def is_full(self) -> bool:
        return self._count == self._capacity

    def __len__(self) -> int:
        return self._count


# ─── Bug 3: Watchdog Kicked by Wrong Thread ───────────────────────────────────
# BUG: a heartbeat thread calls kick() on a fixed timer, independent of the worker.
#   Even if the worker deadlocks, the heartbeat keeps kicking → fault never detected.
#   System appears healthy (no timeout) while doing nothing. Silent mission failure.
# FIX: kick() must be called from the worker itself, AFTER completing real work.

class WatchdogBuggy:
    def __init__(self, timeout_s: float, on_expire):
        self._timeout = timeout_s
        self._on_expire = on_expire
        self._last_kick = time.monotonic()
        self._lock = threading.Lock()
        # BUG: heartbeat kicks unconditionally, masking any worker deadlock
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def kick(self):
        with self._lock:
            self._last_kick = time.monotonic()

    def _heartbeat_loop(self):
        while True:
            self.kick()          # BUG: no connection to whether worker is alive
            time.sleep(0.05)

    def _monitor_loop(self):
        while True:
            time.sleep(0.05)
            with self._lock:
                elapsed = time.monotonic() - self._last_kick
            if elapsed > self._timeout:
                self._on_expire()
                return


class Watchdog:
    """FIX: worker is the only entity that calls kick(), after doing real work."""
    def __init__(self, timeout_s: float, on_expire):
        self._timeout = timeout_s
        self._on_expire = on_expire
        self._last_kick = time.monotonic()
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def kick(self):
        # Call from the actual worker loop after completing meaningful work.
        # Use monotonic clock — wall clock can jump; monotonic never goes backward.
        with self._lock:
            self._last_kick = time.monotonic()

    def stop(self):
        self._running = False

    def _monitor_loop(self):
        while self._running:
            time.sleep(0.05)
            with self._lock:
                elapsed = time.monotonic() - self._last_kick
            if elapsed > self._timeout:
                self._on_expire()
                return


# ─── Bug 4: Race Condition — Check-Then-Act ───────────────────────────────────
# BUG: two threads can both read firing==False before either writes True.
#   Both call _do_fire() → actuator fires twice. Mission safety issue.
# FIX: hold the lock across the entire check-and-set operation.

class ActuatorControllerBuggy:
    def __init__(self):
        self.firing = False  # BUG: plain bool, no lock

    def fire(self) -> bool:
        if not self.firing:       # Thread A checks → False
            self.firing = True    # Thread B also sees False before this executes
            return True           # both threads fire
        return False


class ActuatorController:
    def __init__(self):
        self._lock = threading.Lock()
        self._firing = False

    def fire(self) -> bool:
        # FIX: lock makes check-and-set atomic
        with self._lock:
            if self._firing:
                return False
            self._firing = True
            return True

    def complete(self):
        with self._lock:
            self._firing = False

    @property
    def is_firing(self) -> bool:
        with self._lock:
            return self._firing


# ─── Bug 5: Integer Overflow / Sequence Number Wrap ──────────────────────────
# BUG: sequence number masked to 8 bits → wraps at 255.
#   At 100 Hz: wraps every 2.55 seconds.
#   Ground station sees seq jump 255 → 0 and counts it as 255 lost packets.
# FIX: use 32-bit sequence number. Wraps after ~497 days at 100 Hz.

class TelemetryProducerBuggy:
    def __init__(self):
        self._seq = 0

    def next_seq(self) -> int:
        # BUG: u8 equivalent — wraps at 255
        self._seq = (self._seq + 1) & 0xFF
        return self._seq


class TelemetryProducer:
    def __init__(self):
        self._seq = 0

    def next_seq(self) -> int:
        # FIX: u32 equivalent — wraps after ~4 billion (~497 days at 100 Hz)
        # & 0xFFFFFFFF makes the u32 semantics explicit (Python int is unbounded)
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return self._seq


# ─── Bug 6: Missing Timeout on Blocking Read ─────────────────────────────────
# BUG: serial_port.read(9) with no timeout configured on the port.
#   If IMU power-cycles or drops a byte, read() hangs forever.
#   No exception, no error — entire task silently blocks.
# FIX: set timeout= on Serial open; treat partial read as an error.

class ReadTimeoutError(Exception):
    def __init__(self, got: int, expected: int):
        super().__init__(f"Read timeout: got {got} of {expected} bytes")
        self.got = got
        self.expected = expected


def read_exact_or_timeout(buf: bytes, expected: int) -> bytes:
    """Simulates a properly-wrapped serial read with timeout enforcement."""
    if len(buf) < expected:
        raise ReadTimeoutError(got=len(buf), expected=expected)
    return buf[:expected]

# Real-world pattern:
#   import serial
#   port = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0.1)  # ← timeout here
#   data = port.read(9)       # returns partial bytes on timeout (doesn't block forever)
#   if len(data) != 9:
#       raise ReadTimeoutError(got=len(data), expected=9)


# ─── Bug 7: State Machine — Invalid Transition ───────────────────────────────
# BUG: FAULT → NOMINAL directly. A broken component resumes operation without
#   going through SafeMode checks → hardware running in unknown state.
# FIX: FAULT must → SAFE_MODE first. Only SAFE_MODE → NOMINAL after checks pass.

class FsmState(Enum):
    NOMINAL   = auto()
    FAULT     = auto()
    SAFE_MODE = auto()

class FsmEvent(Enum):
    FAULT_DETECTED = auto()
    SAFE_MODE_CMD  = auto()
    CHECKS_PASSED  = auto()
    RESET          = auto()


BUGGY_TRANSITIONS: dict = {
    (FsmState.NOMINAL,   FsmEvent.FAULT_DETECTED): FsmState.FAULT,
    (FsmState.FAULT,     FsmEvent.CHECKS_PASSED):  FsmState.NOMINAL,  # BUG: skip SafeMode
    (FsmState.SAFE_MODE, FsmEvent.RESET):          FsmState.NOMINAL,
}

FIXED_TRANSITIONS: dict = {
    (FsmState.NOMINAL,   FsmEvent.FAULT_DETECTED): FsmState.FAULT,
    (FsmState.FAULT,     FsmEvent.SAFE_MODE_CMD):  FsmState.SAFE_MODE,
    (FsmState.SAFE_MODE, FsmEvent.CHECKS_PASSED):  FsmState.NOMINAL,
    (FsmState.SAFE_MODE, FsmEvent.RESET):          FsmState.NOMINAL,
    # (FsmState.FAULT, FsmEvent.CHECKS_PASSED) ← REMOVED: illegal direct path
}

class FaultManager:
    def __init__(self, transitions: dict = None):
        self.state = FsmState.NOMINAL
        self._transitions = transitions if transitions is not None else FIXED_TRANSITIONS

    def transition(self, event: FsmEvent) -> FsmState:
        key = (self.state, event)
        if key not in self._transitions:
            raise ValueError(
                f"Invalid transition: {self.state.name} + {event.name}"
            )
        # Log old→new BEFORE updating (post-mortem: you want to know the last good state)
        self.state = self._transitions[key]
        return self.state


# ─── Bug 8: Resource Leak on Error Path ──────────────────────────────────────
# BUG: Device.open() acquires a hardware resource (HW lock, power rail, etc.).
#   If calibrate() raises, the device is never shut down → resource stays held.
#   In FSW: next reboot may fail to acquire the device because it's still "open".
# FIX: explicit try/except cleanup, or use a context manager (__enter__/__exit__).

class Device:
    def __init__(self, name: str):
        self.name = name
        self.calibrated = False
        # Simulate: HW resource acquired on construction

    def calibrate(self):
        if 'fail' in self.name:
            raise RuntimeError("calibration failed")
        self.calibrated = True

    def shutdown(self):
        pass  # release HW resource

    # FIX option 2: context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.shutdown()
        return False  # don't suppress the exception


def open_and_calibrate_buggy(name: str) -> Device:
    dev = Device(name)
    dev.calibrate()  # BUG: raises → dev.shutdown() never called
    return dev


def open_and_calibrate(name: str) -> Device:
    dev = Device(name)
    try:
        dev.calibrate()
    except Exception:
        dev.shutdown()  # FIX: always clean up on error path
        raise
    return dev


def open_and_calibrate_ctx(name: str) -> Device:
    # FIX option 2: context manager is cleaner
    with Device(name) as dev:
        dev.calibrate()
    return dev  # Note: __exit__ only calls shutdown() on exception


# ─── Bug 9: Mutex Deadlock (ABBA) ────────────────────────────────────────────
# BUG: write_data acquires data_lock → file_lock (A → B)
#        rotate_log acquires file_lock → data_lock (B → A)
#   Thread A holds data, waits for file.
#   Thread B holds file, waits for data.
#   → Neither can proceed. System hangs with no exception, no log output.
# FIX: always acquire locks in the same order, everywhere. Or use a single lock.

class DataLoggerBuggy:
    def __init__(self):
        self._data_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._log = []

    def write_data(self, data: str):
        with self._data_lock:          # Thread A: acquires data_lock
            with self._file_lock:      # Thread A: blocks waiting for file_lock
                self._log.append(data)

    def rotate_log(self):
        with self._file_lock:          # Thread B: acquires file_lock
            with self._data_lock:      # Thread B: blocks waiting for data_lock → DEADLOCK
                self._log.clear()


class DataLogger:
    """FIX: single coarse lock eliminates the ordering problem entirely."""
    def __init__(self):
        self._lock = threading.Lock()
        self._log = []

    def write_data(self, data: str):
        with self._lock:
            self._log.append(data)

    def rotate_log(self):
        with self._lock:
            self._log.clear()

    def entry_count(self) -> int:
        with self._lock:
            return len(self._log)


# ─── Bug 10: Spurious Wakeup — if vs while ────────────────────────────────────
# This bug doesn't exist in the Rust version: Rust's Condvar.wait() forces while.
# Python has no such guard — easy to write 'if' and get it subtly wrong.
#
# BUG: uses 'if' on the condition check. A spurious wakeup (OS can wake a waiting
#   thread for any reason) passes the 'if' guard even if the condition is still
#   False → enqueue to a full buffer, or dequeue from an empty one.
# FIX: ALWAYS use 'while' with condition variables. This is a universal rule:
#   Java, Python, C++, Rust all require it (Rust enforces it; others do not).

class BoundedQueueBuggy:
    def __init__(self, capacity: int):
        self._queue = []
        self._capacity = capacity
        self._lock = threading.Lock()
        self._not_full  = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def enqueue(self, item):
        with self._not_full:
            if len(self._queue) >= self._capacity:  # BUG: 'if' — one spurious wakeup passes through
                self._not_full.wait()
            self._queue.append(item)
            self._not_empty.notify()

    def dequeue(self):
        with self._not_empty:
            if len(self._queue) == 0:               # BUG: same
                self._not_empty.wait()
            item = self._queue.pop(0)
            self._not_full.notify()
            return item


class BoundedQueue:
    def __init__(self, capacity: int):
        self._queue = []
        self._capacity = capacity
        self._lock = threading.Lock()
        self._not_full  = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def enqueue(self, item):
        with self._not_full:
            while len(self._queue) >= self._capacity:  # FIX: 'while' re-checks after every wakeup
                self._not_full.wait()
            self._queue.append(item)
            self._not_empty.notify()

    def dequeue(self):
        with self._not_empty:
            while len(self._queue) == 0:               # FIX
                self._not_empty.wait()
            item = self._queue.pop(0)
            self._not_full.notify()
            return item

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)


# ─── Tests ────────────────────────────────────────────────────────────────────
# Run: python -m pytest bug_hunt.py -v

import pytest

# Bug 1 — endianness
def test_endianness_buggy_misreads():
    # 256 = 0x00000100, big-endian bytes: [0x00, 0x00, 0x01, 0x00]
    buf = struct.pack('>i', 256)
    assert parse_sensor_buggy(buf) != 256

def test_endianness_fixed_reads_correctly():
    buf = struct.pack('>i', 256)
    assert parse_sensor_fixed(buf) == 256

def test_endianness_negative_big_endian():
    buf = struct.pack('>i', -1000)
    assert parse_sensor_fixed(buf) == -1000


# Bug 2 — ring buffer
def test_ring_buffer_buggy_panics_after_full_cycle():
    rb = RingBufferBuggy(capacity=4)
    for i in range(4):
        rb.push(i)         # fills: head → 4
    rb.pop()               # count drops to 3, head stays at 4
    with pytest.raises(IndexError):
        rb.push(99)        # data[4] → IndexError

def test_ring_buffer_wraps_correctly():
    rb = RingBuffer(capacity=4)
    assert rb.push(1) and rb.push(2) and rb.push(3) and rb.push(4)
    assert rb.is_full
    assert not rb.push(5)       # full — rejected
    assert rb.pop() == 1
    assert rb.pop() == 2
    assert rb.push(5)           # head wraps here
    assert rb.pop() == 3
    assert rb.pop() == 4
    assert rb.pop() == 5
    assert rb.is_empty

def test_ring_buffer_survives_full_cycle():
    rb = RingBuffer(capacity=256)
    for i in range(256):
        assert rb.push(i % 256)
    for _ in range(256):
        rb.pop()
    assert rb.is_empty
    assert rb.push(42)          # would IndexError in buggy version
    assert rb.pop() == 42


# Bug 4 — race condition (deterministic single-thread check of the logic)
def test_actuator_prevents_double_fire():
    ctrl = ActuatorController()
    assert ctrl.fire()          # accepted
    assert not ctrl.fire()      # rejected — already firing
    ctrl.complete()
    assert ctrl.fire()          # accepted again


# Bug 5 — integer overflow
def test_seq_u8_wraps_at_255():
    prod = TelemetryProducerBuggy()
    for _ in range(255):
        prod.next_seq()
    assert prod.next_seq() == 0   # wrapped — looks like packet loss to ground

def test_seq_u32_does_not_wrap_early():
    prod = TelemetryProducer()
    for _ in range(256):
        prod.next_seq()
    assert prod.next_seq() == 257


# Bug 6 — missing timeout
def test_partial_read_raises_timeout():
    with pytest.raises(ReadTimeoutError) as exc_info:
        read_exact_or_timeout(b'\x00' * 5, 9)
    assert exc_info.value.got == 5
    assert exc_info.value.expected == 9

def test_full_read_succeeds():
    result = read_exact_or_timeout(b'\xAB' * 9, 9)
    assert len(result) == 9


# Bug 7 — state machine
def test_fault_to_nominal_requires_safe_mode():
    fm = FaultManager()
    fm.transition(FsmEvent.FAULT_DETECTED)
    assert fm.state == FsmState.FAULT

    # Direct FAULT → NOMINAL must fail
    with pytest.raises(ValueError):
        fm.transition(FsmEvent.CHECKS_PASSED)
    assert fm.state == FsmState.FAULT  # state must not change on invalid transition

    # Correct path: FAULT → SAFE_MODE → NOMINAL
    fm.transition(FsmEvent.SAFE_MODE_CMD)
    assert fm.state == FsmState.SAFE_MODE
    fm.transition(FsmEvent.CHECKS_PASSED)
    assert fm.state == FsmState.NOMINAL

def test_buggy_fsm_allows_illegal_transition():
    fm_buggy = FaultManager(transitions=BUGGY_TRANSITIONS)
    fm_buggy.transition(FsmEvent.FAULT_DETECTED)
    # BUG: this should fail but doesn't in the buggy version
    fm_buggy.transition(FsmEvent.CHECKS_PASSED)
    assert fm_buggy.state == FsmState.NOMINAL  # jumped straight back — bad!


# Bug 8 — resource leak
def test_buggy_open_raises_without_cleanup():
    # Just confirms it raises; resource leak not observable in unit test
    with pytest.raises(RuntimeError):
        open_and_calibrate_buggy("device_fail")

def test_fixed_open_raises_and_cleans_up():
    with pytest.raises(RuntimeError):
        open_and_calibrate("device_fail")

def test_fixed_open_succeeds():
    dev = open_and_calibrate("device_ok")
    assert dev.calibrated


# Bug 9 — deadlock (single-thread: confirms write/rotate/count work)
def test_data_logger_no_deadlock():
    logger = DataLogger()
    logger.write_data("line 1")
    logger.write_data("line 2")
    assert logger.entry_count() == 2
    logger.rotate_log()
    assert logger.entry_count() == 0
    logger.write_data("line 3")
    assert logger.entry_count() == 1


# Bug 10 — spurious wakeup (structural: 'while' vs 'if')
def test_bounded_queue_basic():
    q = BoundedQueue(capacity=3)
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    assert len(q) == 3
    assert q.dequeue() == 1
    assert q.dequeue() == 2
    q.enqueue(4)
    assert q.dequeue() == 3
    assert q.dequeue() == 4
    assert len(q) == 0
