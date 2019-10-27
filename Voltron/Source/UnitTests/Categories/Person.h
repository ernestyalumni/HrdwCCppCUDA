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

    PersonT(
      std::string name,
      const std::string& surname,
      GenderT gender,
      int age = 0
      ):
      name_{name},
      surname_{surname},
      gender_{gender},
      age_{age}
    {}

    std::string name() const
    {
      return name_;
    }

    std::string surname() const
    {
      return surname_;
    }

    GenderT gender() const
    {
      return gender_;
    }

    int age() const
    {
      return age_;
    }

    void print(
      std::ostream &out,
      PersonT::OutputFormatT format) const
    {
      if (format == PersonT::name_only)
      {
        out << name();
      }
      else if (format == PersonT::full_name)
      {
        out << name() << surname();
      }
    }

  private:

    std::string name_;
    std::string surname_;
    GenderT gender_;
    int age_;
};

} // namespace Categories

#endif // _UNIT_TESTS_CATEGORIES_PERSON_H_