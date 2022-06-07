#include "rapidjson/document.h"
#include <gtest/gtest.h>

using rapidjson::Document;

namespace GoogleUnitTests
{
namespace Dependencies
{
namespace RapidJson
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RapidJsonTests, DocumentDefaultConstructs)
{
  Document document {};

  SUCCEED();
}

//------------------------------------------------------------------------------
/// \url https://github.com/Tencent/rapidjson/blob/master/example/tutorial/tutorial.cpp
//------------------------------------------------------------------------------
TEST(RapidJsonTests, PassesBasicTutorial)
{
  const char json[] {
    " { \"hello\" : \"world\", \"t\" : true , \"f\" : false, \"n\": null, \"i\":123, \"pi\": 3.1416, \"a\":[1, 2, 3, 4] } "};

  Document document {};
  EXPECT_FALSE(document.Parse(json).HasParseError());

  // Access values in document.

  // Document is a JSON value that represent the root of the DOM. Root can be
  // either an object or array.
  EXPECT_TRUE(document.IsObject());
}

} // namespace RapidJson
} // namespace Dependencies
} // namespace GoogleUnitTests
