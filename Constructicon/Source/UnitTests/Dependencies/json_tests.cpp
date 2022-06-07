#include "nlohmann/json.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using json = nlohmann::json;

namespace GoogleUnitTests
{
namespace Dependencies
{
namespace Json
{

//------------------------------------------------------------------------------
/// \url https://github.com/nlohmann/json#examples
//------------------------------------------------------------------------------
TEST(JsonTests, ConstructsFromStringLiteral)
{
  json j = "{ \"happy\": true, \"pi\": 3.141 }"_json;

  EXPECT_TRUE(j["happy"]);
  EXPECT_DOUBLE_EQ(j["pi"], 3.141);

  // Or even nicer with a raw string literal.
  auto j2 = R"(
    {
      "happy": true,
      "pi": 3.141
    }
  )"_json;

  EXPECT_TRUE(j2["happy"]);
  EXPECT_DOUBLE_EQ(j2["pi"], 3.141);
}

//------------------------------------------------------------------------------
TEST(JsonTests, LoadsJsonFileFromOperator)
{
  // cf. https://api.nasa.gov/
  // cf. https://api.nasa.gov/insight_weather/?api_key=DEMO_KEY&feedtype=json&ver=1.0
  auto path = std::filesystem::current_path();
  path /= "../Source/UnitTests/TestData";
  path /= "MarsInsight-20220607.json";
  std::ifstream in {path};

  // cf. https://stackoverflow.com/questions/33628250/c-reading-a-json-object-from-file-with-nlohmann-json

  json jf = json::parse(in);
  auto at_iterator = jf["validity_checks"]["1219"]["AT"]["sol_hours_with_data"].begin();
  int expected {6};
  while (at_iterator != jf["validity_checks"]["1219"]["AT"]["sol_hours_with_data"].end())
  {
    EXPECT_EQ(*at_iterator, expected);
    ++expected;
    ++at_iterator;
  }

  EXPECT_TRUE(jf["validity_checks"]["1219"]["AT"].contains("valid"));
  EXPECT_FALSE(jf["validity_checks"]["1219"]["AT"]["valid"].get<bool>());

  auto hws_iterator = jf["validity_checks"]["1219"]["HWS"]["sol_hours_with_data"].begin();
  expected = 6;
  while (hws_iterator != jf["validity_checks"]["1219"]["HWS"]["sol_hours_with_data"].end())
  {
    EXPECT_EQ(*hws_iterator, expected);
    ++expected;
    ++hws_iterator;
  }

  EXPECT_FALSE(jf["validity_checks"]["1219"]["HWS"]["valid"].get<bool>());
}

} // namespace Json
} // namespace Dependencies
} // namespace GoogleUnitTests
