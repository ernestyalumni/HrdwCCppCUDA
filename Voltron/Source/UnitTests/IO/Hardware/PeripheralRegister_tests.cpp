#include "IO/Hardware/PeripheralRegister.h"

#include <boost/test/unit_test.hpp>
#include <cstdint>

using IO::Hardware::PeripheralRegister;

BOOST_AUTO_TEST_SUITE(IO)
BOOST_AUTO_TEST_SUITE(Hardware)
BOOST_AUTO_TEST_SUITE(PeripheralRegister_tests)

//------------------------------------------------------------------------------
/// Test fixture: Simulates hardware registers using static memory
/// In real hardware, these would be actual memory-mapped addresses
//------------------------------------------------------------------------------
struct PeripheralRegisterFixture
{
  // Simulated hardware registers in memory
  static inline uint8_t simulated_status_reg_ {0x00};
  static inline uint8_t simulated_control_reg_ {0x00};
  
  PeripheralRegisterFixture()
  {
    // Reset simulated hardware state before each test
    simulated_status_reg_ = 0x00;
    simulated_control_reg_ = 0x00;
  }
  
  // Helper to map addresses to simulated registers
  static volatile uint8_t* get_simulated_reg(uint32_t address)
  {
    if (address == 0x40000000)
    {
      return &simulated_status_reg_;
    }
    if (address == 0x40000001)
    {
      return &simulated_control_reg_;
    }
    return nullptr;
  }
};

//------------------------------------------------------------------------------
/// Test 1: Volatile Prevents Caching - Hardware Can Change Values
/// 
/// KEY POINT: Without volatile, compiler might cache register value.
/// With volatile, each read is an actual memory access, so we see hardware
///changes.
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  VolatilePreventsCaching_HardwareCanChangeValues,
  PeripheralRegisterFixture)
{
  PeripheralRegister status_reg(&simulated_status_reg_);
  
  // Initial state: hardware sets register to 0x00
  simulated_status_reg_ = 0x00;
  BOOST_TEST(status_reg.read() == 0x00);
  
  // CRITICAL: Hardware changes register value (e.g., interrupt handler, DMA,
  //etc.)
  // Without volatile, compiler might cache the first read and never check
  // again!
  simulated_status_reg_ = 0x42;
  
  // With volatile, this read MUST access actual memory and see the change
  uint8_t value {status_reg.read()};
  BOOST_TEST(value == 0x42, "Volatile ensures we see hardware changes");
  
  // Multiple reads - each must be actual memory access
  simulated_status_reg_ = 0xAB;
  uint8_t read1 {status_reg.read()};
  // Hardware changes it again
  simulated_status_reg_ = 0xCD;
  uint8_t read2 {status_reg.read()};
  
  BOOST_TEST(read1 == 0xAB);
  BOOST_TEST(read2 == 0xCD, "Each read sees current hardware state");
  BOOST_TEST(
    read1 != read2,
    "Without volatile, compiler might optimize to cached value");
  
  // Write operations also must be actual memory writes
  status_reg.write_register(0x99);
  BOOST_TEST(simulated_status_reg_ == 0x99, "Write actually occurs in memory");
}

//------------------------------------------------------------------------------
/// Test 2: Status Polling Loop - Why Volatile is Critical
/// 
/// KEY POINT: Without volatile, compiler might optimize polling loop to infinite loop!
/// This is the MOST COMMON embedded bug when volatile is forgotten.
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(StatusPollingLoop_VolatilePreventsInfiniteLoop, PeripheralRegisterFixture)
{
  PeripheralRegister status_reg(&simulated_status_reg_);
  
  // Simulate hardware that sets READY bit (bit 0) after some time
  // Initially not ready
  simulated_status_reg_ = 0x00;

  // CRITICAL: This polling loop MUST check memory on each iteration
  // Without volatile, compiler might optimize to:
  //   uint8_t cached = *register;  // Cache once
  //   while (cached & 0x01 == 0) {}  // INFINITE LOOP - never checks again!
  //
  // With volatile, compiler MUST generate actual memory read each iteration
  int iterations {0};
  const int MAX_ITERATIONS {1000};
  const uint8_t READY_BIT {0x01};
  
  while ((status_reg.read() & READY_BIT) == 0 && iterations < MAX_ITERATIONS)
  {
    iterations++;
    
    // Simulate hardware eventually setting ready bit (e.g., after 50
    // iterations)
    if (iterations == 50)
    {
      simulated_status_reg_ |= READY_BIT;  // Hardware sets ready
    }
  }
  
  // Loop should exit when hardware sets ready bit
  BOOST_TEST(iterations == 50, "Loop exits when hardware sets ready bit");
  BOOST_TEST(
    (status_reg.read() & READY_BIT) != 0,
    "Status register shows ready");

  // Without volatile, this test would hang because compiler would optimize
  // the loop to check a cached value that never changes
}

//------------------------------------------------------------------------------
/// Test 3: Read-Modify-Write Operations - Volatile Preserves Operation Sequence
/// 
/// KEY POINT: Volatile ensures read-modify-write operations occur in sequence.
/// Each operation (read, modify, write) must be actual memory access.
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(
  ReadModifyWriteOperations_VolatilePreservesSequence,
  PeripheralRegisterFixture)
{
  PeripheralRegister ctrl_reg(&simulated_control_reg_);
  
  // Initial state
  simulated_control_reg_ = 0x0F;
  BOOST_TEST(ctrl_reg.read() == 0x0F);
  
  // Set bits: Read current value, OR with mask, Write back
  // volatile ensures: actual read, actual OR, actual write
  ctrl_reg.set_bits(0x30);
  BOOST_TEST(simulated_control_reg_ == 0x3F, "0x0F | 0x30 = 0x3F");
  
  // Clear bits: Read current value, AND with inverted mask, Write back
  // volatile ensures each step is actual memory operation
  ctrl_reg.clear_bits(0x0C);
  BOOST_TEST(simulated_control_reg_ == 0x33, "0x3F & ~0x0C = 0x33");

  // Toggle bits: Read current value, XOR with mask, Write back
  ctrl_reg.toggle_bits(0x11);
  BOOST_TEST(simulated_control_reg_ == 0x22, "0x33 ^ 0x11 = 0x22");

  // Demonstrate that hardware can change register between operations
  // Hardware modifies register
  simulated_control_reg_ = 0xAA;
  // Our operation must read current value (0xAA) 
  ctrl_reg.set_bits(0x55);
  BOOST_TEST(
    simulated_control_reg_ == 0xFF,
    "0xAA | 0x55 = 0xFF - volatile ensures we read 0xAA, not cached value");
}

BOOST_AUTO_TEST_SUITE_END() // PeripheralRegister_tests
BOOST_AUTO_TEST_SUITE_END() // Hardware
BOOST_AUTO_TEST_SUITE_END() // IO
