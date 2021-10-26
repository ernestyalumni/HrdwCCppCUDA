#include "Utilities/AsBits.h"

#include <cstdint>
#include <gtest/gtest.h>

using Utilities::AsBits;

namespace GoogleUnitTests
{
namespace Utilities
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AsBitsTests, DefaultConstructs)
{
  AsBits<uint8_t> b;

  EXPECT_EQ(b.to_ulong(), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AsBitsTests, ConstructsWithInteger)
{
  AsBits<uint8_t> b {1};

  EXPECT_EQ(b.to_ulong(), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AsBitsTests, ConstructsWithHexValue)
{
  AsBits<uint8_t> b {0x1F};

  EXPECT_EQ(b.to_ulong(), 31);
  EXPECT_EQ(b.as_hex_string(), "1f");
  EXPECT_EQ(b.to_string(), "00011111");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AsBitsTests, HandlesBoolsAsIntegers)
{
  AsBits<bool> t {true};

  EXPECT_EQ(t.to_ulong(), 1);
  EXPECT_EQ(t.as_hex_string(), "01");
  EXPECT_EQ(t.to_string(), "00000001");
  EXPECT_EQ(t.to_integer(), true);

  AsBits<bool> f {false};

  EXPECT_EQ(f.to_ulong(), 0);
  EXPECT_EQ(f.as_hex_string(), "00");
  EXPECT_EQ(f.to_string(), "00000000");
  EXPECT_EQ(f.to_integer(), false);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AsBitsTests, ShowsBitShiftedValues)
{
  AsBits<int32_t> x {1};

  EXPECT_EQ(x.to_ulong(), 1);
  EXPECT_EQ(x.as_hex_string(), "00000001");
  EXPECT_EQ(x.to_string(), "00000000000000000000000000000001");

  AsBits<int32_t> y {1 << (sizeof(int32_t) - 1)};

  EXPECT_EQ(y.to_ulong(), 8);
  EXPECT_EQ(y.as_hex_string(), "00000008");
  EXPECT_EQ(y.to_string(), "00000000000000000000000000001000");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AsBitsTests, ShowsComplementNegatedValues)
{
  {
    // This shows that the expression ( (N & ((~N) + 1)) != N) checks for powers
    // of 2.
    // cf. D.W. Harder, ece250.h.

    int16_t x_int {1};

    AsBits<int16_t> x {static_cast<uint16_t>(x_int)};
    EXPECT_EQ(x.to_ulong(), 1);
    EXPECT_EQ(x.as_hex_string(), "0001");
    EXPECT_EQ(x.to_string(), "0000000000000001");

    EXPECT_EQ((~x).to_ulong(), 65534);
    EXPECT_EQ((~x).to_string(), "1111111111111110");

    x_int = (~x_int) + 1;
    EXPECT_EQ(x_int, -1);
    AsBits<int16_t> y {static_cast<uint16_t>(x_int)};

    EXPECT_EQ(y.to_ulong(), 65535);
    EXPECT_EQ(y.to_string(), "1111111111111111");

    x_int = 2;
    x_int = (~x_int) + 1;
    EXPECT_EQ(x_int, -2);

    x_int = 3;
    x_int = (~x_int) + 1;
    EXPECT_EQ(x_int, -3);

    x_int = 4;
    x_int = (~x_int) + 1;
    EXPECT_EQ(x_int, -4);
  }

}

} // namespace Utilities
} // namespace GoogleUnitTests
