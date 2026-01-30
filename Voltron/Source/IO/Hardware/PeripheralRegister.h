#ifndef IO_HARDWARE_PERIPHERAL_REGISTER_H
#define IO_HARDWARE_PERIPHERAL_REGISTER_H

#include <cstdint>

namespace IO
{
namespace Hardware
{

//------------------------------------------------------------------------------
/// Example: Memory-mapped peripheral register access using volatile
/// 
/// volatile prevents compiler optimization for externally changing variables
/// (e.g., hardware registers, interrupt-modified variables)
/// 
/// Key points:
/// - Compiler cannot cache reads/writes
/// - Each access generates actual memory operation
/// - Critical for hardware registers that change outside program control
//------------------------------------------------------------------------------

class PeripheralRegister
{
  public:

    PeripheralRegister(const uint32_t address = 0x40000000):
      address_{address},
      // register_ uses the value in address_ as a memory address. It points to
      // address (e.g. 0x40000000) in the system's memory space. It does not
      // point to where address_ is stored.
      // register_ uses the VALUE (address=0x40000000) as an address.
      register_{reinterpret_cast<volatile uint8_t*>(address_) }
    {}

    // Constructor for testing (accepts pointer directly)
    PeripheralRegister(volatile uint8_t* reg_ptr):
      // Not used when pointer is provided
      address_{0},
      register_{reg_ptr}
    {}

    uint32_t address_;
    // This points to 1 byte at that address, address_.
    volatile uint8_t* const register_;

    // Memory Address Space:
    // ┌─────────────────────────────────────┐
    // │ 0x00000000  │  RAM (your program)   │
    // │ ...         │                        │
    // │ 0x3FFFFFFF  │                        │
    // ├─────────────────────────────────────┤
    // │ 0x40000000  │ ← Hardware Register   │ ← register_ points HERE
    // │             │   (1 byte: 0x00-0xFF) │
    // │ 0x40000001  │   Another register    │
    // │ 0x40000002  │   Another register    │
    // │ ...         │                        │
    // └─────────────────────────────────────┘

    inline volatile uint8_t* const get_register()
    {
      return reinterpret_cast<volatile uint8_t*>(address_);
    }

    inline void write_register(const uint8_t value)
    {
      *register_ = value;
    }

    inline uint8_t read() const
    {
      return *register_;
    }

    // Read-Modify-Write operations

    inline void set_bits(const uint8_t mask) const
    {
      // 01 | 11 = 11
      *register_ |= mask;
    }

    inline void clear_bits(const uint8_t mask) const
    {
      // 01 & 00 = 00
      *register_ &= ~mask;
    }

    inline void toggle_bits(const uint8_t mask) const
    {
      // 01 ^ 11 = 10
      *register_ ^= mask;
    }
};

// Hypothetical memory-mapped peripheral register at address 0x40000000
// volatile: value may change outside program control (hardware)
// const pointer: address never changes, but value pointed to can change
volatile uint8_t* const PERIPHERAL_REG = 
    reinterpret_cast<volatile uint8_t*>(0x40000000);

// Alternative: register as a reference (C++ style)
inline volatile uint8_t& get_peripheral_reg()
{
    return *reinterpret_cast<volatile uint8_t*>(0x40000000);
}

// Write to peripheral register
inline void write_reg(uint8_t value)
{
    *PERIPHERAL_REG = value;  // Compiler cannot optimize this away
}

// Read from peripheral register
inline uint8_t read_reg()
{
    return *PERIPHERAL_REG;  // Compiler must generate actual read
}

// Example: Status register with bit fields
struct StatusRegister
{
    static constexpr uint8_t READY_BIT = 0x01;
    static constexpr uint8_t ERROR_BIT = 0x02;
    static constexpr uint8_t BUSY_BIT = 0x04;
    
    static volatile uint8_t* const STATUS_REG;
    
    static bool is_ready()
    {
        return (*STATUS_REG & READY_BIT) != 0;
    }
    
    static bool has_error()
    {
        return (*STATUS_REG & ERROR_BIT) != 0;
    }
    
    static void wait_until_ready()
    {
        // volatile ensures each read is actually performed
        while ((*STATUS_REG & READY_BIT) == 0)
        {
            // Busy wait - compiler cannot optimize this loop away
        }
    }
};

// Example: Control register
struct ControlRegister
{
    static volatile uint8_t* const CTRL_REG;
    
    static void enable()
    {
        *CTRL_REG |= 0x01;  // Set bit 0
    }
    
    static void disable()
    {
        *CTRL_REG &= ~0x01;  // Clear bit 0
    }
};

} // namespace Hardware
} // namespace IO

#endif // IO_HARDWARE_PERIPHERAL_REGISTER_H