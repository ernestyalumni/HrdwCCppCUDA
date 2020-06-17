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

bool Contains::operator()(const string& substring)
{
  return (given_string_.find(substring) > -1);
}

bool Contains::operator()(const exception& e)
{
  return operator()(e.what());
}

} // namespace Tools
} // namespace UnitTests