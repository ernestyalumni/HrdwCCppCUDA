//------------------------------------------------------------------------------
/// \file ReaderMonad_tests.cpp
/// \ref Ivan Čukić, Functional Programming in C++,  Manning Publications;
/// 1st edition (November 19, 2018). ISBN-13: 978-1617293818
//------------------------------------------------------------------------------
#include "Categories/Monads/ReaderMonad.h"

#include <boost/test/unit_test.hpp>
#include <cassert>
#include <cstdio> // std::tmpfile
#include <filesystem> // std::filesystem::temp_directory_path;
#include <fstream> 
#include <iostream>
#include <optional>
#include <regex>
#include <string>

using Categories::Monads::ReaderMonad::Local;
using Categories::Monads::ReaderMonad::MultiplicationEndomorphismComposed;
using Categories::Monads::ReaderMonad::Unit;
using Categories::Monads::ReaderMonad::apply_morphism;
using Categories::Monads::ReaderMonad::ask;
using Categories::Monads::ReaderMonad::bind;
using Categories::Monads::ReaderMonad::bind_;
using Categories::Monads::ReaderMonad::return_;
using Categories::Monads::ReaderMonad::unit;

BOOST_AUTO_TEST_SUITE(Categories)
BOOST_AUTO_TEST_SUITE(Monads)
BOOST_AUTO_TEST_SUITE(ReaderMonad_tests)

class TemporaryDirectory
{
  public:

    TemporaryDirectory(const std::string& temporary_directory):
      temporary_directory_{temporary_directory}
    {
      std::filesystem::create_directory(temporary_directory);
    }

    class TemporaryFile
    {
      public:

        TemporaryFile(
          const std::filesystem::path directory,
          const std::string& filename
          ):
          filename_path_{directory / filename}
        {}

        std::filesystem::path filename_path() const
        {
          return filename_path_;
        }

      private:

        std::filesystem::path filename_path_;
    };

    ~TemporaryDirectory()
    {
      std::filesystem::remove_all(temporary_directory_);
    }

    TemporaryFile temporary_file(const std::string& filename)
    {
      return TemporaryFile{temporary_directory_, filename};
    }

  private:

    std::filesystem::path temporary_directory_;
};

// cf. http://www.cplusplus.com/doc/tutorial/files/
// https://stackoverflow.com/questions/3379956/how-to-create-a-temporary-directory-in-c


void set_fake_state(const std::filesystem::path& filepath, const int value)
{
  std::ofstream output_stream {filepath, std::ios::binary};
  output_stream.seekp(0, std::ios::beg);

  output_stream << std::to_string(value) << "\n";

  output_stream.close();
}

std::optional<int> read_fake_state(const std::filesystem::path& filepath)
{
  std::ifstream input_stream {filepath, std::ios::binary};
  std::string s;
  input_stream.seekg(0, std::ios::beg);

  std::getline(input_stream, s);

  if (s.empty())
  {
    return std::nullopt;
  }

  return std::make_optional<int>(std::stoi(s));
}

struct TestEnvironment
{
  bool yes_;
  double x_;
  std::string s_;
};

TestEnvironment initial_test_environment {false, 42.0, "Start"};
const TestEnvironment test_environment {true, 1.616, "Middle"};

// Test morphisms.

double r(const TestEnvironment& environment)
{
  if (environment.yes_)
  {
    return environment.x_;
  }
  return 0.0;
}

std::string s(const TestEnvironment& environment)
{
  if (environment.yes_)
  {
    return environment.s_;
  }
  return "Go back.";
}

// test_morphisms belongs to Hom(E, X).
// Example from https://nbviewer.jupyter.org/github/dbrattli/OSlash/blob/master/notebooks/Reader.ipynb

// Auxiliary function to test morphism.
std::string transform_text(const std::string text)
{
  return std::regex_replace(text, std::regex("Hi"), "Hello");
}

// Serves as a morphism in (E \to X)
auto test_string_morphism = [](const std::string& name) -> std::string
{
  return "Hi " + name + "!";
};

// Serves as a morphism f : X \to T(Y) that will have an endomorphism applied to
// it.
auto another_string_morphism(const std::string& text)
{
  const std::string replaced_text {transform_text(text)};

  return unit<std::string, std::string>(replaced_text);
}

// Modified example.
auto test_double_string_morphism = [](const double value) -> std::string
{
  return "Value : " + std::to_string(value) + ", ";
};

BOOST_AUTO_TEST_SUITE(Morphisms_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestMorphismsMapEnvironmentToCategoryObjectType)
{
  BOOST_TEST(r(initial_test_environment) == 0.0);
  BOOST_TEST(r(test_environment) == 1.616);
  BOOST_TEST(s(initial_test_environment) == "Go back.");
  BOOST_TEST(s(test_environment) == "Middle");  
  BOOST_TEST(test_string_morphism("Jean Talon") == "Hi Jean Talon!");
  BOOST_TEST(test_double_string_morphism(2.998) == "Value : 2.998000, ");
  BOOST_TEST(transform_text("Hi there") == "Hello there");
  BOOST_TEST(another_string_morphism("Hi Hi Hi you")("42") ==
    "Hello Hello Hello you");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(UnitReturnsConstantFunctions)
{
  {
    auto unit_component = unit<double, TestEnvironment>(69.0);
    BOOST_TEST(unit_component(initial_test_environment) == 69.0);
  }
  {
    auto unit_component = unit<double, TestEnvironment>(1.616);
    BOOST_TEST(unit_component(initial_test_environment) == 1.616);
  }
  // Example from https://nbviewer.jupyter.org/github/dbrattli/OSlash/blob/master/notebooks/Reader.ipynb
  {
    auto unit_component = unit<int, std::string>(42);
    BOOST_TEST(unit_component("Ignored") == 42);
  }
  {
    auto unit_component = unit<std::string, std::string>("Hello there");
    BOOST_TEST(unit_component("Bonjour") == "Hello there");
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(LambdaVersionOfUnitReturnsConstantFunctions)
{
  const auto unit_component = return_(10.0);
  BOOST_TEST(unit_component(initial_test_environment) == 10.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AskReturnsEnvironmentAsIdentity)
{
  const auto result = ask(initial_test_environment);
  BOOST_TEST(!result.yes_);
  BOOST_TEST(result.x_ == 42.0);
  BOOST_TEST(result.s_ == "Start");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ApplyMorphismReturnsValueOfTypeX)
{
  {
    const auto result = apply_morphism(r, initial_test_environment);
    BOOST_TEST(result == 0.0);
  }
  {
    const auto result = apply_morphism(r, test_environment);
    BOOST_TEST(result == 1.616);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Morphisms_tests

BOOST_AUTO_TEST_SUITE(Bind_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TestingStepsInBind)
{
  const std::string test_environment_string {"Sieur de la Salle"};
  {
    auto f_on_X =
      another_string_morphism(test_string_morphism(test_environment_string));
    BOOST_TEST(f_on_X(test_environment_string) == "Hello Sieur de la Salle!");
  }
  {
    auto f_on_X = another_string_morphism(
      apply_morphism(test_string_morphism, test_environment_string));

    BOOST_TEST(apply_morphism(f_on_X, test_environment_string) ==
      "Hello Sieur de la Salle!");
  }
}

// TODO: Make bind work with templates.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindWorksOnStringMorphisms)
{
  auto result = bind(test_string_morphism, another_string_morphism);
  //result("Le Royer");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MultiplicationEndomorphismComposedWorksOnStringMorphisms)
{
  // \mu_Y \circ Tf : [E, X] \to [E, Y]
  // another_string_morphism = f: std::string -> [str, str]
  // test_string_morphism, [E,X], (str -> str)
  // test_string_morphism is a typical morphism. But, f, or another_string_
  // morphism is how to "configure" the output behavior of test_string_morphism.

  auto mu_Tf = MultiplicationEndomorphismComposed{another_string_morphism};
  auto result = mu_Tf(test_string_morphism);
  BOOST_TEST(result("Le Royer") == "Hello Le Royer!");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(BindAsLambdaWorksOnStringMorphisms)
{
  auto result = bind_(test_string_morphism, another_string_morphism);

  BOOST_TEST(result("Le Royer") == "Hello Le Royer!");
}

BOOST_AUTO_TEST_SUITE_END() // Bind_tests

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TemporaryDirectoryAndFStreamMakesReadWriteEnvironment)
{
  {
    std::filesystem::path temporary_path {
      std::filesystem::temp_directory_path()};

    std::string temporary_directory {"TestValues"};

    std::string temporary_file_name {"TestValue"};
    //std::filesystem::path new_path {temporary_path /= temporary_file_name};
    std::filesystem::path new_path {temporary_path /= temporary_directory};

    //std::cout << temporary_file_name << " " << new_path << "\n";

    std::ofstream ofs {new_path / temporary_file_name, std::ios::binary};
   
    ofs << "1" << "\n";

    //ofs.close();

    // read back
    std::ifstream istream {new_path / temporary_file_name, std::ios::binary};

    std::string s;

    // Doesnt work.
    //istream >> s;

    std::getline(istream, s);

    //std::cout << " s : " << s << std::endl;
    //int s_int {std::stoi(s)};

    //BOOST_TEST(s_int == 1);

    //BOOST_TEST(s == "1");

    // Remove the directory and its contents.
    std::filesystem::remove_all(new_path);
  }
  {
    std::ofstream ofs {"TestValueTest", std::ios::binary};
    ofs << "1" << "\n";
    ofs.close();

    std::ifstream istream {"TestValueTest", std::ios::binary};

    std::string s;

    std::getline(istream, s);

    //istream >> s;

    //std::cout << " s : " << s << "\n";

    int s_int {std::stoi(s)};

    BOOST_TEST(s_int == 1);
  }
  {
    
    std::filesystem::path temporary_directory {"TestValues"};
    std::filesystem::create_directory(temporary_directory);

    std::string temporary_file_name {"TestValue"};
    std::filesystem::path new_path {temporary_directory / temporary_file_name};

    //std::cout << new_path << "\n";

    std::ofstream ofs {new_path, std::ios::binary};
    ofs << "1" << "\n";
    ofs.close();

    std::ifstream istream {new_path, std::ios::binary};

    std::string s;

    std::getline(istream, s);

    BOOST_TEST((!s.empty()));

    //std::cout << " s : " << s << "\n";

    int s_int {std::stoi(s)};

    BOOST_TEST(s_int == 1); 

    istream.close();

    ofs.open(new_path, std::ios::binary);

    ofs.seekp(0, std::ios::beg);
    ofs << "0" << "\n";
    ofs.close();

    istream.open(new_path, std::ios::binary);

    istream.seekg(0, std::ios::beg);

    std::getline(istream, s);

    s_int = std::stoi(s);

    BOOST_TEST(s_int == 0); 

    std::filesystem::remove_all(temporary_directory); 
  }
  {
    TemporaryDirectory temp_dir {"TestValues"};
    auto temp_file = temp_dir.temporary_file("TestValue");

    set_fake_state(temp_file.filename_path(), 1);

    auto fake_state = read_fake_state(temp_file.filename_path());

    BOOST_TEST(static_cast<bool>(fake_state));
    BOOST_TEST(*fake_state == 1);

    set_fake_state(temp_file.filename_path(), 0);

    fake_state = read_fake_state(temp_file.filename_path());

    BOOST_TEST(static_cast<bool>(fake_state));
    BOOST_TEST(*fake_state == 0);

  }
}


BOOST_AUTO_TEST_SUITE_END() // ReaderMonad_tests
BOOST_AUTO_TEST_SUITE_END() // Monads
BOOST_AUTO_TEST_SUITE_END() // Categories