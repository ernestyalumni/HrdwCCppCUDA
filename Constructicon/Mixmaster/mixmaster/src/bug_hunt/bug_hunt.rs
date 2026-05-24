// Bug Hunt Practice — FSW Context (Matter Intelligence interview prep)
//
// For each bug: read the code, cover the "FIX" section, find it yourself first.
// Target: < 5 min per bug. Verbalize: what you see → runtime symptom → fix → FSW impact.

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Bug 1: Endianness (Hardware → Software) ─────────────────────────────────
// Sensor sends 4-byte big-endian i32 over UART. ICD says: big-endian.
// Symptom: garbage values — looks like sensor noise, it's a software bug.

pub struct EndiannessBug;

impl EndiannessBug {
    // BUG: from_le_bytes when hardware sends big-endian
    pub fn parse_buggy(buf: &[u8; 4]) -> i32 {
        i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
    }

    // FIX: match the hardware ICD — always check endianness at every HW/SW boundary
    pub fn parse_fixed(buf: &[u8; 4]) -> i32 {
        i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
    }
}

// ─── Bug 2: Off-by-one in Ring Buffer ────────────────────────────────────────
// 1 Hz=1 cycle/sec, 100 Hz=100 cycles/sec, T=10ms period
// Symptom: works for first N pushes, then OOB panic. At 100 Hz this is 2.56s.

pub struct RingBufferBuggy {
    data: [u8; 256],
    head: usize,
    count: usize,
}

impl RingBufferBuggy {
    //---------------------------------------------------------------------------------------------
    /// e.g. N=3 (instead of 256)
    /// push(A),push(B),push(C)
    /// head=3, count=3. Head kept climbing, no wrap.
    /// pop(): count=2, 3-2-1 = 0 is the tail that gets popped. head stays at 3.
    /// push(D): count=2 < 3, data[3]: index 3 doesn't exist, panic
    //---------------------------------------------------------------------------------------------
    pub fn new() -> Self { Self { data: [0; 256], head: 0, count: 0 } }

    // BUG: head grows monotonically — panics when head reaches 256 after a pop frees space
    pub fn push(&mut self, val: u8) -> bool {
        if self.count >= 256 { return false; }
        self.data[self.head] = val;
        self.head += 1; // ← missing % 256: head sticks at 256 after a full cycle
        self.count += 1;
        true
    }

    // pop decrements count but head never moves back → after one full cycle,
    // the next push writes to data[256] → OOB panic
    pub fn pop(&mut self) -> Option<u8> {
        if self.count == 0 { return None; }
        self.count -= 1;
        Some(self.data[(self.head - self.count - 1) % 256])
    }
}

pub struct RingBuffer<const N: usize> {
    data: [u8; N],
    head: usize,
    count: usize,
}

impl<const N: usize> RingBuffer<N> {
    pub fn new() -> Self { Self { data: [0; N], head: 0, count: 0 } }

    // FIX: wrap head modulo N (use & (N-1) if N is always a power of 2)
    pub fn push(&mut self, val: u8) -> bool {
        if self.count >= N { return false; }
        self.data[self.head] = val;
        self.head = (self.head + 1) % N;
        self.count += 1;
        true
    }

    pub fn pop(&mut self) -> Option<u8> {
        //------------------------------------------------------------------------------------------
        /// tail is count slots behind head, but usize can't go negative, so to keep the number
        /// positive, write head + N - count
        /// e.g. 2 + 3 - 3 = 2, 1 + 3 - 2 = 2
        //------------------------------------------------------------------------------------------        
        if self.count == 0 { return None; }
        let tail = (self.head + N - self.count) % N;
        self.count -= 1;
        Some(self.data[tail])
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == N }
}

// ─── Bug 3: Watchdog Kicked by Wrong Thread ───────────────────────────────────
// BUG pattern: a heartbeat thread calls kick() on a timer, independent of the worker.
//   Even if the worker deadlocks, heartbeat keeps kicking → fault never detected.
// FIX: kick() must be called from the worker itself, after completing real work.

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub struct Watchdog {
    last_kick_ms: Arc<AtomicU64>,
    timeout_ms: u64,
}

impl Watchdog {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            last_kick_ms: Arc::new(AtomicU64::new(now_ms())),
            timeout_ms,
        }
    }

    // Call from the actual worker loop after completing meaningful work — NOT from a heartbeat
    pub fn kick(&self) {
        self.last_kick_ms.store(now_ms(), Ordering::Relaxed);
    }

    pub fn is_expired(&self) -> bool {
        now_ms().saturating_sub(self.last_kick_ms.load(Ordering::Relaxed)) > self.timeout_ms
    }
}

// ─── Bug 4: Race Condition — Check-Then-Act ──────────────────────────────────
// BUG pattern (plain bool): if !self.firing { self.firing = true; self.do_fire(); }
//   Two threads can both read false before either writes true → double-fire.
//   In FSW, double-firing a thruster or actuator is a mission safety issue.

pub struct ActuatorController {
    firing: Arc<AtomicBool>,
}

impl ActuatorController {
    pub fn new() -> Self { Self { firing: Arc::new(AtomicBool::new(false)) } }

    // FIX: compare_exchange atomically checks-and-sets — only one thread wins
    pub fn fire(&self) -> bool {
        self.firing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn complete(&self) {
        self.firing.store(false, Ordering::Release);
    }

    pub fn is_firing(&self) -> bool {
        self.firing.load(Ordering::Acquire)
    }
}

// ─── Bug 5: Integer Overflow / Sequence Number Wrap ──────────────────────────
// BUG: u8 wraps at 255. At 100 Hz that's every 2.55 seconds.
//   Ground station sees seq jump 255 → 0 and counts it as packet loss.

pub struct TelemetryProducerBuggy { seq: u8 }

impl TelemetryProducerBuggy {
    pub fn new() -> Self { Self { seq: 0 } }
    // BUG: u8 wraps at 255. wrapping_add models release-mode behavior;
    //   plain += also panics in debug builds — both are wrong for a seq counter.
    pub fn next_seq(&mut self) -> u8 { self.seq = self.seq.wrapping_add(1); self.seq }
}

pub struct TelemetryProducer { seq: u32 }

impl TelemetryProducer {
    pub fn new() -> Self { Self { seq: 0 } }
    // FIX: u32 wraps after ~4 billion (497 days at 100 Hz).
    //   wrapping_add makes the intent explicit — no accidental overflow in debug builds.
    pub fn next_seq(&mut self) -> u32 { self.seq = self.seq.wrapping_add(1); self.seq }
}

// ─── Bug 6: Missing Timeout on Blocking Read ─────────────────────────────────
// BUG: serial_port.read(9) with no timeout → hangs forever if sensor drops.
//   Entire FSW task blocks. No exception, just silence. Very hard to diagnose.
// FIX: set timeout on port open; treat partial reads as TimeoutError.
//
// (Actual serial I/O requires external crate; this captures the validation contract.)

#[derive(Debug, PartialEq)]
pub enum ReadError {
    Timeout { got: usize, expected: usize },
}

pub fn read_exact_or_timeout(buf: &[u8], expected: usize) -> Result<&[u8], ReadError> {
    if buf.len() < expected {
        Err(ReadError::Timeout { got: buf.len(), expected })
    } else {
        Ok(&buf[..expected])
    }
}

// ─── Bug 7: State Machine — Invalid Transition ───────────────────────────────
// BUG: FAULT → NOMINAL directly allowed, skipping SafeMode checks.
//   A broken component resumes operation without verification.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FsmState { Nominal, Fault, SafeMode }

#[derive(Debug)]
pub enum FsmEvent { FaultDetected, SafeModeCmd, ChecksPassed, Reset }

#[derive(Debug)]
pub struct InvalidTransition { pub from: FsmState, pub event: &'static str }

pub struct FaultManager { state: FsmState }

impl FaultManager {
    pub fn new() -> Self { Self { state: FsmState::Nominal } }
    pub fn state(&self) -> FsmState { self.state }

    // FIX: no direct Fault → Nominal path; must go Fault → SafeMode → Nominal
    pub fn transition(&mut self, event: FsmEvent) -> Result<FsmState, InvalidTransition> {
        let next = match (self.state, &event) {
            (FsmState::Nominal,  FsmEvent::FaultDetected) => FsmState::Fault,
            (FsmState::Fault,    FsmEvent::SafeModeCmd)   => FsmState::SafeMode,
            (FsmState::SafeMode, FsmEvent::ChecksPassed)  => FsmState::Nominal,
            (FsmState::SafeMode, FsmEvent::Reset)          => FsmState::Nominal,
            // All other transitions are illegal
            _ => {
                let event_name = match event {
                    FsmEvent::FaultDetected => "FaultDetected",
                    FsmEvent::SafeModeCmd   => "SafeModeCmd",
                    FsmEvent::ChecksPassed  => "ChecksPassed",
                    FsmEvent::Reset         => "Reset",
                };
                return Err(InvalidTransition { from: self.state, event: event_name });
            }
        };
        // Log old→new before transitioning (post-mortem visibility)
        self.state = next;
        Ok(next)
    }
}

// ─── Bug 8: Resource Leak on Error Path ──────────────────────────────────────
// BUG: Device::open() acquires hardware resources. If calibrate() fails, those
//   resources (HW lock, power rail) stay held because there's no cleanup on the
//   error path. In Rust, File closes via Drop, but custom HW resources do not.
// FIX: RAII guard — implement Drop to call shutdown() if not explicitly released.

pub struct Device { pub name: String, pub calibrated: bool }

impl Device {
    pub fn open(name: &str) -> Result<Self, String> {
        Ok(Self { name: name.to_string(), calibrated: false })
    }
    pub fn calibrate(&mut self) -> Result<(), String> {
        if self.name.contains("fail") { return Err("calibration failed".into()); }
        self.calibrated = true;
        Ok(())
    }
    pub fn shutdown(&mut self) { /* release HW resources */ }
}

// Option<Device> is required because Rust forbids moving out of a type that
// implements Drop. take() extracts the Device without violating that invariant.
pub struct DeviceGuard {
    device: Option<Device>,
    cleanup_on_drop: bool,
}

impl DeviceGuard {
    pub fn new(device: Device) -> Self {
        Self { device: Some(device), cleanup_on_drop: true }
    }

    pub fn device_mut(&mut self) -> &mut Device {
        self.device.as_mut().unwrap()
    }

    // Caller takes ownership — guard disarms its cleanup
    pub fn release(mut self) -> Device {
        self.cleanup_on_drop = false;
        self.device.take().unwrap()
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            if let Some(mut dev) = self.device.take() {
                dev.shutdown();
            }
        }
    }
}

pub fn open_and_calibrate(name: &str) -> Result<Device, String> {
    let mut guard = DeviceGuard::new(Device::open(name)?);
    guard.device_mut().calibrate()?; // failure here → guard drops → shutdown() called
    Ok(guard.release())
}

// ─── Bug 9: Mutex Deadlock (ABBA) ────────────────────────────────────────────
// BUG:
//   write_data:  lock(data_mutex) → lock(file_mutex)
//   rotate_log:  lock(file_mutex) → lock(data_mutex)
//   Thread A holds data, waits for file. Thread B holds file, waits for data. Deadlock.
// FIX: always acquire locks in the same order everywhere. Or use a single lock.

pub struct DataLogger {
    // FIX: single coarse lock eliminates the ordering problem entirely.
    // If you need two locks, document the canonical order and enforce it in code review.
    inner: Arc<Mutex<DataLoggerInner>>,
}

struct DataLoggerInner {
    log: Vec<String>,
    #[allow(dead_code)]
    file_path: String,
}

impl DataLogger {
    pub fn new(file_path: &str) -> Self {
        Self {
            inner: Arc::new(Mutex::new(DataLoggerInner {
                log: Vec::new(),
                file_path: file_path.to_string(),
            })),
        }
    }

    pub fn write_data(&self, data: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.log.push(data.to_string());
        // flush to inner.file_path (simulated)
    }

    pub fn rotate_log(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.log.clear();
        // open new file at inner.file_path (simulated)
    }

    pub fn entry_count(&self) -> usize {
        self.inner.lock().unwrap().log.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Bug 1 — endianness
    // 2^8 = 2^4 * 2^4 = 1 byte
    // 256 = 2^8
    // 0x00 00 01 00
    // 256 (0x00000100) arrives over wire (big-endian, most significant byte comes first) as:
    // buf[0]=0x00 buf[1]=0x00 buf[2]=0x01 buf[3]=0x00
    #[test]
    fn endianness_buggy_misreads_value() {
        let buf: [u8; 4] = [0x00, 0x00, 0x01, 0x00]; // big-endian 256
        assert_ne!(EndiannessBug::parse_buggy(&buf), 256);
    }

    #[test]
    fn endianness_fixed_reads_correct_value() {
        let buf: [u8; 4] = [0x00, 0x00, 0x01, 0x00]; // big-endian 256
        assert_eq!(EndiannessBug::parse_fixed(&buf), 256);
    }

    #[test]
    fn endianness_negative_value_big_endian() {
        let val: i32 = -1000;
        let buf = val.to_be_bytes();
        assert_eq!(EndiannessBug::parse_fixed(&buf), -1000);
    }

    // Bug 2 — ring buffer (buggy version panics after pop+push crosses the 256 boundary)
    // Scenario: fill buffer (head → 256), pop one slot free, push again → data[256] OOB
    #[test]
    #[should_panic]
    fn ring_buffer_buggy_panics_after_full_cycle() {
        let mut rb = RingBufferBuggy::new();
        for i in 0..256u32 { rb.push(i as u8); } // fills buffer; head == 256
        rb.pop();       // count drops to 255, but head stays at 256
        rb.push(99);    // count=255 < 256 passes the guard; data[256] → OOB panic
    }

    #[test]
    fn ring_buffer_wraps_correctly() {
        let mut rb = RingBuffer::<4>::new();
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert!(rb.push(4));
        assert!(rb.is_full());
        assert!(!rb.push(5)); // full — rejected
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert!(rb.push(5)); // head wraps around here
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
        assert_eq!(rb.pop(), Some(5));
        assert!(rb.is_empty());
    }

    #[test]
    fn ring_buffer_survives_full_cycle_256() {
        let mut rb = RingBuffer::<256>::new();
        for i in 0..256u32 { assert!(rb.push(i as u8)); }
        for _ in 0..256 { rb.pop(); }
        assert!(rb.is_empty());
        assert!(rb.push(42)); // would panic in buggy version (head == 256)
        assert_eq!(rb.pop(), Some(42));
    }

    // Bug 4 — race condition / check-then-act
    #[test]
    fn actuator_prevents_double_fire() {
        let ctrl = ActuatorController::new();
        assert!(ctrl.fire());   // accepted
        assert!(!ctrl.fire());  // rejected — already firing
        ctrl.complete();
        assert!(ctrl.fire());   // accepted again after completion
    }

    // Bug 5 — integer overflow
    #[test]
    fn seq_u8_wraps_at_255() {
        let mut prod = TelemetryProducerBuggy::new();
        for _ in 0..255 { prod.next_seq(); }
        assert_eq!(prod.next_seq(), 0); // wrapped — ground station sees gap
    }

    #[test]
    fn seq_u32_does_not_wrap_early() {
        let mut prod = TelemetryProducer::new();
        for _ in 0..256 { prod.next_seq(); }
        assert_eq!(prod.next_seq(), 257); // no wrap
    }

    // Bug 6 — missing timeout
    #[test]
    fn partial_read_returns_timeout_error() {
        let buf = &[0u8; 5];
        let result = read_exact_or_timeout(buf, 9);
        assert_eq!(result, Err(ReadError::Timeout { got: 5, expected: 9 }));
    }

    #[test]
    fn full_read_succeeds() {
        let buf = &[0xABu8; 9];
        let result = read_exact_or_timeout(buf, 9);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 9);
    }

    // Bug 7 — invalid state transition
    #[test]
    fn fault_to_nominal_requires_safe_mode_path() {
        let mut fm = FaultManager::new();
        fm.transition(FsmEvent::FaultDetected).unwrap();
        assert_eq!(fm.state(), FsmState::Fault);

        // Direct Fault → Nominal must fail — state unchanged
        assert!(fm.transition(FsmEvent::ChecksPassed).is_err());
        assert_eq!(fm.state(), FsmState::Fault);

        // Correct path: Fault → SafeMode → Nominal
        fm.transition(FsmEvent::SafeModeCmd).unwrap();
        assert_eq!(fm.state(), FsmState::SafeMode);
        fm.transition(FsmEvent::ChecksPassed).unwrap();
        assert_eq!(fm.state(), FsmState::Nominal);
    }

    #[test]
    fn nominal_to_fault_via_reset_is_invalid() {
        let mut fm = FaultManager::new();
        assert!(fm.transition(FsmEvent::Reset).is_err());
        assert_eq!(fm.state(), FsmState::Nominal);
    }

    // Bug 8 — resource leak / RAII guard
    #[test]
    fn guard_shuts_down_device_on_calibration_failure() {
        // "fail" in name → calibrate() returns Err → guard.drop() → shutdown()
        let result = open_and_calibrate("device_fail");
        assert!(result.is_err());
        // no resource leak: DeviceGuard::drop() called shutdown() before returning
    }

    #[test]
    fn guard_releases_device_on_success() {
        let device = open_and_calibrate("device_ok").unwrap();
        assert!(device.calibrated);
    }

    // Bug 9 — deadlock
    #[test]
    fn data_logger_write_and_rotate_do_not_deadlock() {
        let logger = DataLogger::new("/tmp/test.log");
        logger.write_data("line 1");
        logger.write_data("line 2");
        assert_eq!(logger.entry_count(), 2);
        logger.rotate_log();
        assert_eq!(logger.entry_count(), 0);
        logger.write_data("line 3");
        assert_eq!(logger.entry_count(), 1);
    }
}
