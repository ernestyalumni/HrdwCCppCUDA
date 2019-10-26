//------------------------------------------------------------------------------
/// \file Person.h
/// \author 
/// \brief 
/// \ref https://gitlab.com/manning-fpcpp-book/code-examples/blob/master/common/person.h
///-----------------------------------------------------------------------------

#ifndef _UNIT_TESTS_CATEGORIES_PERSON_H_
#define _UNIT_TESTS_CATEGORIES_PERSON_H_

#include <string>

namespace Categories
{

class PersonT
{
  public:

    enum GenderT
    {
      female,
      male
    };

    enum OutputFormatT
    {
      name_only,
      full_name
    };

    PersonT() :
      name_{"John"},
      surname_{"Doe"},
      gender_{male}
    {}

    std::string name() const
    {
      return name_;
    }

  private:

    std::string name_;
    std::string surname_;
    GenderT gender_;
    int age_;
};

} // namespace Categories

#endif // _UNIT_TESTS_CATEGORIES_PERSON_H_