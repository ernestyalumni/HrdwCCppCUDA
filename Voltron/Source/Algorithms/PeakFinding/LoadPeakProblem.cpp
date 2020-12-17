//------------------------------------------------------------------------------
/// \file LoadPeakProblem.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Load the 2-dim. problem for peak finding of Problem Set 1, MIT 6.006.
/// \details MIT OCW 6.006, Fall 2011, Problem Set 1.
//------------------------------------------------------------------------------
#include "LoadPeakProblem.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using std::make_optional;
using std::nullopt;
using std::optional;
using std::size_t;
using std::string;

namespace Algorithms
{
namespace PeakFinding
{

LoadPeakProblem::LoadPeakProblem() = default;

optional<size_t> LoadPeakProblem::parse_first_equal_sign(const string& input_line)
{
	const size_t result {input_line.find('=')};
	return result == string::npos ? nullopt : make_optional<size_t>(result);
}

} // namespace PeakFinding
} // namespace Algorithms
