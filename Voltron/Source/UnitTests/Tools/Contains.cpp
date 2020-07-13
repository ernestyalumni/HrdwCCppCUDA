#include "Contains.h"

#include <exception>
#include <string>

using std::exception;
using std::string;

namespace UnitTests
{
namespace Tools
{
  
Contains::Contains(const string& given_string):
  given_string_{given_string}
{}

bool Contains::operator()(const string& input_string)
{
//  return (given_string_.find(substring) > -1);
  return (input_string.find(given_string_) > -1);
}

bool Contains::operator()(const exception& e)
{
  return operator()(e.what());
}

} // namespace Tools
} // namespace UnitTests