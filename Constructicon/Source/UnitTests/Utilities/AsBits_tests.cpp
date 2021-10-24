#include "Utilities/AsBits.h"

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

} // namespace Utilities
} // namespace GoogleUnitTests
