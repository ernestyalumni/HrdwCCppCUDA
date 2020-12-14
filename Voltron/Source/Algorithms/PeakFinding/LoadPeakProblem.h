//------------------------------------------------------------------------------
/// \file LoadPeakProblem.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Load the 2-dim. problem for peak finding of Problem Set 1, MIT 6.006.
/// \details MIT OCW 6.006, Fall 2011, Problem Set 1.
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H
#define ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H

#include <cstdint>
#include <vector>

namespace Algorithms
{
namespace PeakFinding
{

class LoadPeakProblem
{
  public:

    LoadPeakProblem();

  protected:

    //parse_equal_sign


  private:

    bool first_left_bracket_found_{false};
};

} // namespace PeakFinding
} // namespace Algorithms

#endif // ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H
