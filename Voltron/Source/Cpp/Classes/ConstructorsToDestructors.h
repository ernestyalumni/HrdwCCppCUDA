//------------------------------------------------------------------------------
/// \file RuleOf5.h
/// \author Ernest Yeung
/// \brief Classes demonstrating copy and move constructors, assignments,
///   destructors.
///-----------------------------------------------------------------------------
#ifndef CPP_CLASSES_CONSTRUCTORS_TO_DESTRUCTORS_H
#define CPP_CLASSES_CONSTRUCTORS_TO_DESTRUCTORS_H

#include <string>

namespace Cpp
{
namespace Classes
{

class DefaultConstructs
{
  public:

    //--------------------------------------------------------------------------
    /// \details Classes that have custom ctors, dtors
    //--------------------------------------------------------------------------


    //--------------------------------------------------------------------------
    /// \brief Default Constructor.
    /// \details
    //--------------------------------------------------------------------------
    DefaultConstructs();

  private:

    std::string s_data_;
    int int_data_;
    bool is_default_constructed_{false};
};

} // namespace Classes
} // namespace Cpp

#endif // CPP_CLASSES_CONSTRUCTORS_TO_DESTRUCTORS_H