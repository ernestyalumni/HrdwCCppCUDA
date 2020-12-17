//------------------------------------------------------------------------------
/// \file LoadPeakProblem.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Load the 2-dim. problem for peak finding of Problem Set 1, MIT 6.006.
/// \details MIT OCW 6.006, Fall 2011, Problem Set 1.
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H
#define ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H

#include <cstddef> // std::size_t
#include <optional>
#include <string>

namespace Algorithms
{
namespace PeakFinding
{

class LoadPeakProblem
{
  public:

    LoadPeakProblem();

    bool first_equal_sign_found() const
    {
    	return first_equal_sign_found_;
    }

  protected:

    //parse_equal_sign

  	std::optional<std::size_t> parse_first_equal_sign(const std::string& input_line);

  private:

  	bool first_equal_sign_found_ {false};
    bool first_left_bracket_found_{false};
};

} // namespace PeakFinding
} // namespace Algorithms

#endif // ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H
