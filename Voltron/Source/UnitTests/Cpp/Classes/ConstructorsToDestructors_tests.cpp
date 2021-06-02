#include "Cpp/Classes/ConstructorsToDestructors.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <sstream> // std::stringstream
#include <streambuf> // std::streambuf
#include <string>
#include <utility>

using Cpp::Classes::CustomDestructorEncapsulated;
using Cpp::Classes::CustomDestructorLight;
using Cpp::Classes::DefaultConstructs;
using Cpp::Classes::MoveOnlyLight;
using Cpp::Classes::NoDefaultConstruction;
using Cpp::Classes::return_no_default_construction_as_type;
using Cpp::Classes::return_rvalue_move_only_light;
using std::clog;
using std::move;
using std::streambuf;
using std::string;
using std::stringstream;
using std::to_string;

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Classes)
BOOST_AUTO_TEST_SUITE(ConstructorsToDestructors_tests)

struct TestClogSetup
{
  stringstream stream_;
  streambuf* original_clog_;

  TestClogSetup():
    stream_{},
    original_clog_{clog.rdbuf(stream_.rdbuf())}
  {}

  ~TestClogSetup()
  {
    // Restore std::clog
    clog.rdbuf(original_clog_);
  }
};

BOOST_AUTO_TEST_SUITE(DefaultConstructs_tests)

int default_constructs_default_ctor_counter_start_value {
  DefaultConstructs::default_ctor_counter()};

int default_constructs_dtor_counter_start_value {
  DefaultConstructs::dtor_counter()};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StaticValuesStartAtZeroBeforeAnyObjects)
{
  // TODO: Could possibly break if these classes with static variables are
  // "started up" elsewhere.
  BOOST_TEST(DefaultConstructs::default_ctor_counter() == 0);
  BOOST_TEST(DefaultConstructs::dtor_counter() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsDefaultConstructs)
{
  //----------------------------------------------------------------------------
  /// \ref https://en.cppreference.com/w/cpp/io/basic_stringstream
  /// \details Implements input and output operations on string based streams.
  /// Effectively stores instance of std::basic_string and performs input and
  /// output operations on it.
  /// We can read from a string as if it were a stream like cin.
  //----------------------------------------------------------------------------
  stringstream stream;

  //----------------------------------------------------------------------------
  /// \ref https://en.cppreference.com/w/cpp/io/basic_streambuf
  /// \details stringstream.rdbuf returns pointer to underlying raw string
  /// device object.
  /// std::basic_streambuf controls input and output to a character sequence. It
  /// includes and provides access to
  /// 1. Controlled character sequence, aka buffer, which may contain input
  /// sequence (aka get area) for buffering input operations and/or output
  /// sequence (aka put area) for buffering output operations.
  /// 2. Associated character sequence (aka source (for input) or sink (for
  /// output)). This maybe an entity that's accessed through OS API (file, TCP
  /// socket, serial port, other character device), or it may be an object that
  /// can be interpreted as a character source or sink.
  ///
  /// cf. https://www.cplusplus.com/reference/streambuf/streambuf/
  /// A stream buffer is an object in charge of performing reading and writing
  /// operations of the stream object it's associated with.
  ///
  /// protected:
  /// basic_streambuf(const basic_streambuf& rhs);
  ///
  /// Constructs copy of rhs, initializes streambuf's 6 ptrs and locale object
  /// with copies of values held by rhs. Shallow copy.
  ///
  /// ios::rdbuf() returns ptr to stream buffer object currently associated with
  /// the stream.
  /// ios::rdbuf(streambuf* sb); sets object pointed by sb as stream buffer
  /// associated with stream and clears error state flags.
  //----------------------------------------------------------------------------
  streambuf* original_clog {clog.rdbuf(stream.rdbuf())};

  DefaultConstructs a;

  BOOST_TEST(DefaultConstructs::default_ctor_counter() == 1);
  BOOST_TEST(DefaultConstructs::dtor_counter() == 0);

  BOOST_TEST(stream.str() == "DefaultConstructs default constructs:1");

  // Restore original clog.
  clog.rdbuf(original_clog);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(AccessorReturningReferenceCanMutateMember,
  TestClogSetup)
{
  DefaultConstructs a;

  BOOST_TEST(DefaultConstructs::default_ctor_counter() ==
    default_constructs_default_ctor_counter_start_value + 2);
  BOOST_TEST(DefaultConstructs::dtor_counter() ==
    default_constructs_dtor_counter_start_value + 1);

  BOOST_TEST(stream_.str() == "DefaultConstructs default constructs:" +
    to_string(default_constructs_default_ctor_counter_start_value + 2)); 

  BOOST_TEST(a.s_data() == "");

  string& str_ref {a.s_data()};
  string str_var {"DefaultConstructString"};

  str_ref = str_var;
  BOOST_TEST(a.s_data() == "DefaultConstructString");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(AccessorReturningConstPassedAsRvalue, TestClogSetup)
{
  DefaultConstructs a;

  BOOST_TEST(DefaultConstructs::default_ctor_counter() ==
    default_constructs_default_ctor_counter_start_value + 3);
  BOOST_TEST(DefaultConstructs::dtor_counter() ==
    default_constructs_dtor_counter_start_value + 2);

  BOOST_TEST(stream_.str() == "DefaultConstructs default constructs:" +
    to_string(default_constructs_default_ctor_counter_start_value + 3));

  a.int_data(42);

  // cannot bind non-const lvalue reference of type 'T&' to rvalue of type 'T'
  //int& int_ref {a.int_data()};

  int int_var {a.int_data()};
  BOOST_TEST(int_var == 42);
  int_var += 69;

  BOOST_TEST_REQUIRE(int_var == 111);
  BOOST_TEST(a.int_data() == 42);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(AccessorReturningTypePassedAsRvalue, TestClogSetup)
{
  DefaultConstructs a;

  BOOST_TEST(DefaultConstructs::default_ctor_counter() ==
    default_constructs_default_ctor_counter_start_value + 4);
  BOOST_TEST(DefaultConstructs::dtor_counter() ==
    default_constructs_dtor_counter_start_value + 3);

  BOOST_TEST(stream_.str() == "DefaultConstructs default constructs:" +
    to_string(default_constructs_default_ctor_counter_start_value + 4));

  BOOST_TEST(a.is_default_constructed());

  bool bool_var {a.is_default_constructed()};

  bool_var = false;

  BOOST_TEST_REQUIRE(!bool_var);
  BOOST_TEST(a.is_default_constructed());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ObjectDestroyedPastScope, TestClogSetup)
{
  {
    DefaultConstructs a;

    BOOST_TEST(DefaultConstructs::default_ctor_counter() ==
      default_constructs_default_ctor_counter_start_value + 5);
    BOOST_TEST(DefaultConstructs::dtor_counter() ==
      default_constructs_dtor_counter_start_value + 4);

    BOOST_TEST(stream_.str() == "DefaultConstructs default constructs:" +
      to_string(default_constructs_default_ctor_counter_start_value + 5));

    BOOST_TEST(a.is_default_constructed());

    //--------------------------------------------------------------------------
    /// \ref https://stackoverflow.com/questions/20731/how-do-you-clear-a-stringstream-variable
    /// https://en.cppreference.com/w/cpp/io/basic_stringstream/swap
    /// Exchanges state of stream with those of other.
    //--------------------------------------------------------------------------
    stringstream().swap(stream_); 
  }

  // TODO: Investigate if not needed, to call .sync().
  // Synchronizes with underlying storage device.
  stream_.sync();

  BOOST_TEST(DefaultConstructs::dtor_counter() ==
    default_constructs_dtor_counter_start_value + 5);

  BOOST_TEST(stream_.str() == "DefaultConstructs destructs:" +
    to_string(default_constructs_dtor_counter_start_value + 5));
}

BOOST_AUTO_TEST_SUITE_END() // DefaultConstructs_tests

BOOST_AUTO_TEST_SUITE(NoDefaultConstruction_tests)

int no_default_construction_ctor_1_counter_start_value {
  NoDefaultConstruction::ctor_1_counter()};

int no_default_construction_dtor_counter_start_value {
  NoDefaultConstruction::dtor_counter()};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(ConstructsWithMutableReference, TestClogSetup)
{
  BOOST_TEST(NoDefaultConstruction::ctor_1_counter() == 0);
  BOOST_TEST(NoDefaultConstruction::dtor_counter() == 0);

  int int_var {42};

  NoDefaultConstruction a {int_var, 69};

  BOOST_TEST(NoDefaultConstruction::ctor_1_counter() ==
    no_default_construction_ctor_1_counter_start_value + 1);

  BOOST_TEST(stream_.str() == "NoDefaultConstruction constructs #1:" +
    to_string(no_default_construction_ctor_1_counter_start_value + 1));

  a.increment_int_ref(3);

  BOOST_TEST(a.int_ref() == 45);
  BOOST_TEST(int_var == 45);

  BOOST_TEST(a.int_data() == 69);

  int& int_ref {a.int_data()};

  int_ref += 5;

  BOOST_TEST(a.int_data() == 74);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(LocalVariableInFunctionScopeDoesNotGetDestroyed,
  TestClogSetup)
{
  BOOST_TEST(NoDefaultConstruction::ctor_1_counter() ==
    no_default_construction_ctor_1_counter_start_value + 1);
  BOOST_TEST(NoDefaultConstruction::dtor_counter() ==
    no_default_construction_dtor_counter_start_value + 1);

  int int_var {42};
  int int_value {69};

  NoDefaultConstruction a {
    return_no_default_construction_as_type(int_var, int_value)};

  BOOST_TEST(NoDefaultConstruction::ctor_1_counter() ==
    no_default_construction_ctor_1_counter_start_value + 2);
  BOOST_TEST(NoDefaultConstruction::dtor_counter() ==
    no_default_construction_dtor_counter_start_value + 1);

  BOOST_TEST(stream_.str() ==
    "NoDefaultConstruction constructs #1:" +
    to_string(no_default_construction_ctor_1_counter_start_value + 2));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(LocalVariableGetsDestroyedOutsideScope, TestClogSetup)
{
  BOOST_TEST(NoDefaultConstruction::ctor_1_counter() ==
    no_default_construction_ctor_1_counter_start_value + 2);
  BOOST_TEST(NoDefaultConstruction::dtor_counter() ==
    no_default_construction_dtor_counter_start_value + 2);

  int int_var {42};

  {
    NoDefaultConstruction a {
      return_no_default_construction_as_type(int_var, 69)};

    BOOST_TEST(NoDefaultConstruction::ctor_1_counter() ==
      no_default_construction_ctor_1_counter_start_value + 3);
    BOOST_TEST(NoDefaultConstruction::dtor_counter() ==
      no_default_construction_dtor_counter_start_value + 2);

    BOOST_TEST(stream_.str() ==
      "NoDefaultConstruction constructs #1:" +
      to_string(no_default_construction_ctor_1_counter_start_value + 3));

    stringstream().swap(stream_); 
  }

  BOOST_TEST(NoDefaultConstruction::ctor_1_counter() ==
    no_default_construction_ctor_1_counter_start_value + 3);
  BOOST_TEST(NoDefaultConstruction::dtor_counter() ==
    no_default_construction_dtor_counter_start_value + 3);

  BOOST_TEST(stream_.str() ==
    "NoDefaultConstruction destructs:" +
    to_string(no_default_construction_dtor_counter_start_value + 3));
}

BOOST_AUTO_TEST_SUITE_END() // NoDefaultConstruction_tests

BOOST_AUTO_TEST_SUITE(MoveOnlyLight_tests)

int move_only_light_ctor_counter_start_value {MoveOnlyLight::ctor_counter()};
int move_only_light_move_ctor_counter_start_value {
  MoveOnlyLight::move_ctor_counter()};
int move_only_light_move_assign_counter_start_value {
  MoveOnlyLight::move_assign_counter()};
int move_only_light_dtor_counter_start_value {MoveOnlyLight::dtor_counter()};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(MoveConstructs, TestClogSetup)
{
  BOOST_TEST(MoveOnlyLight::ctor_counter() ==
    move_only_light_ctor_counter_start_value);
  BOOST_TEST(MoveOnlyLight::move_ctor_counter() ==
    move_only_light_move_ctor_counter_start_value);
  BOOST_TEST(MoveOnlyLight::move_assign_counter() ==
    move_only_light_move_assign_counter_start_value);
  BOOST_TEST(MoveOnlyLight::dtor_counter() ==
    move_only_light_dtor_counter_start_value);

  {
    MoveOnlyLight a {42};

    BOOST_TEST(MoveOnlyLight::ctor_counter() ==
      move_only_light_ctor_counter_start_value + 1);
    BOOST_TEST(MoveOnlyLight::move_ctor_counter() ==
      move_only_light_move_ctor_counter_start_value);
    BOOST_TEST(MoveOnlyLight::move_assign_counter() ==
      move_only_light_move_assign_counter_start_value);
    BOOST_TEST(MoveOnlyLight::dtor_counter() ==
      move_only_light_dtor_counter_start_value);

    BOOST_TEST(stream_.str() ==
      "MoveOnlyLight constructs:" +
      to_string(move_only_light_ctor_counter_start_value + 1));

    stringstream().swap(stream_); 

    BOOST_TEST(a.data() == 42);

    MoveOnlyLight b {move(a)};

    BOOST_TEST(MoveOnlyLight::ctor_counter() ==
      move_only_light_ctor_counter_start_value + 1);
    BOOST_TEST(MoveOnlyLight::move_ctor_counter() ==
      move_only_light_move_ctor_counter_start_value + 1);
    BOOST_TEST(MoveOnlyLight::move_assign_counter() ==
      move_only_light_move_assign_counter_start_value);
    BOOST_TEST(MoveOnlyLight::dtor_counter() ==
      move_only_light_dtor_counter_start_value);

    BOOST_TEST(stream_.str() ==
      "MoveOnlyLight move constructs:" +
      to_string(move_only_light_move_ctor_counter_start_value + 1) +
      "data_:42other.data_:-1");

    BOOST_TEST(b.data() == 42);

    BOOST_TEST(a.data() == -1);

    stringstream().swap(stream_); 
  }

  BOOST_TEST(MoveOnlyLight::ctor_counter() ==
    move_only_light_ctor_counter_start_value + 1);
  BOOST_TEST(MoveOnlyLight::move_ctor_counter() ==
    move_only_light_move_ctor_counter_start_value + 1);
  BOOST_TEST(MoveOnlyLight::move_assign_counter() ==
    move_only_light_move_assign_counter_start_value);
  BOOST_TEST(MoveOnlyLight::dtor_counter() ==
    move_only_light_dtor_counter_start_value + 2);

  BOOST_TEST(stream_.str() ==
    "MoveOnlyLight destructs:" +
    to_string(move_only_light_dtor_counter_start_value + 1) +
    "data_:42" +
    "MoveOnlyLight destructs for data_:-1" +
    "MoveOnlyLight destructs:" +
    to_string(move_only_light_dtor_counter_start_value + 2) +
    "data_:-1"
  );
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// This also tests for the fact that "Moved Local Variable is Destructed
// Outside of Function Scope."
BOOST_FIXTURE_TEST_CASE(FunctionCanReturnRvalue, TestClogSetup)
{
  BOOST_TEST(MoveOnlyLight::ctor_counter() ==
    move_only_light_ctor_counter_start_value + 1);
  BOOST_TEST(MoveOnlyLight::move_ctor_counter() ==
    move_only_light_move_ctor_counter_start_value + 1);
  BOOST_TEST(MoveOnlyLight::move_assign_counter() ==
    move_only_light_move_assign_counter_start_value);
  BOOST_TEST(MoveOnlyLight::dtor_counter() ==
    move_only_light_dtor_counter_start_value + 2);

  {
    stringstream().swap(stream_); 

    MoveOnlyLight a {return_rvalue_move_only_light(42)};

    BOOST_TEST(MoveOnlyLight::ctor_counter() ==
      move_only_light_ctor_counter_start_value + 2);
    BOOST_TEST(MoveOnlyLight::move_ctor_counter() ==
      move_only_light_move_ctor_counter_start_value + 2);
    BOOST_TEST(MoveOnlyLight::move_assign_counter() ==
      move_only_light_move_assign_counter_start_value);
    BOOST_TEST(MoveOnlyLight::dtor_counter() ==
      move_only_light_dtor_counter_start_value + 3);

    // Moved local variable is destroyed.
    BOOST_TEST(stream_.str() ==
      "MoveOnlyLight constructs:" +
      to_string(move_only_light_ctor_counter_start_value + 2) +
      "MoveOnlyLight move constructs:" +
      to_string(move_only_light_move_ctor_counter_start_value + 2) +
      "data_:42other.data_:-1" +
      "MoveOnlyLight destructs for data_:-1" +
      "MoveOnlyLight destructs:" +
      to_string(move_only_light_dtor_counter_start_value + 3) +
      "data_:-1");

    stringstream().swap(stream_); 

    BOOST_TEST(a.data() == 42);
  }

  BOOST_TEST(MoveOnlyLight::ctor_counter() ==
    move_only_light_ctor_counter_start_value + 2);
  BOOST_TEST(MoveOnlyLight::move_ctor_counter() ==
    move_only_light_move_ctor_counter_start_value + 2);
  BOOST_TEST(MoveOnlyLight::move_assign_counter() ==
    move_only_light_move_assign_counter_start_value);
  BOOST_TEST(MoveOnlyLight::dtor_counter() ==
    move_only_light_dtor_counter_start_value + 4);

  BOOST_TEST(stream_.str() ==
    "MoveOnlyLight destructs:" +
    to_string(move_only_light_dtor_counter_start_value + 4) +
    "data_:42");
}

BOOST_AUTO_TEST_SUITE_END() // MoveOnlyLight_tests

BOOST_AUTO_TEST_SUITE(CustomDestructorEncapsulated_tests)

int custom_destructor_light_move_ctor_counter_start_value {
  CustomDestructorLight::move_ctor_counter()};
int custom_destructor_light_dtor_counter_start_value {
  CustomDestructorLight::dtor_counter()};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(DestructorInvokedWhenOutsideScope, TestClogSetup)
{
  BOOST_TEST(CustomDestructorLight::move_ctor_counter() ==
    custom_destructor_light_move_ctor_counter_start_value);
  BOOST_TEST(CustomDestructorLight::dtor_counter() ==
    custom_destructor_light_dtor_counter_start_value);

  int int_var {42};

  {
    CustomDestructorLight a {int_var};

    BOOST_TEST(CustomDestructorLight::dtor_counter() ==
      custom_destructor_light_dtor_counter_start_value);

    BOOST_TEST(stream_.str() == "");

    stringstream().swap(stream_); 
  }

  BOOST_TEST(CustomDestructorLight::dtor_counter() ==
    custom_destructor_light_dtor_counter_start_value + 1);

  BOOST_TEST(stream_.str() ==
    "CustomDestructorLight destructs:" +
    to_string(custom_destructor_light_dtor_counter_start_value + 1) +
    "data_:" +
    to_string(int_var));
}

int custom_destructor_encapsulated_dtor_counter_start_value {
  CustomDestructorEncapsulated::dtor_counter()};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_CASE(DestructorDestroysEncapsulatedObjectInOrder,
  TestClogSetup)
{
  BOOST_TEST(CustomDestructorEncapsulated::dtor_counter() ==
    custom_destructor_encapsulated_dtor_counter_start_value);

  BOOST_TEST(CustomDestructorLight::dtor_counter() ==
    custom_destructor_light_dtor_counter_start_value + 1);

  int int_var {42};
  int int_var2 {43};

  {
    CustomDestructorEncapsulated a {int_var, int_var2};

    BOOST_TEST(CustomDestructorEncapsulated::dtor_counter() ==
      custom_destructor_encapsulated_dtor_counter_start_value);

    BOOST_TEST(CustomDestructorLight::dtor_counter() ==
      custom_destructor_light_dtor_counter_start_value + 1);

    BOOST_TEST(stream_.str() == "");

    stringstream().swap(stream_); 
  }

  BOOST_TEST(CustomDestructorEncapsulated::dtor_counter() ==
    custom_destructor_encapsulated_dtor_counter_start_value + 1);

  BOOST_TEST(CustomDestructorLight::dtor_counter() ==
    custom_destructor_light_dtor_counter_start_value + 2);

  BOOST_TEST(stream_.str() ==
    "CustomDestructorEncapsulated destructs:" +
    to_string(custom_destructor_encapsulated_dtor_counter_start_value + 1) +
    "data_:" +
    to_string(int_var) +
    " or " +
    to_string(int_var2) +
    "CustomDestructorLight destructs:" +
    to_string(custom_destructor_light_dtor_counter_start_value + 2) +
    "data_:" +
    to_string(int_var));
}

BOOST_AUTO_TEST_SUITE_END() // CustomDestructorEncapsulated_tests

BOOST_AUTO_TEST_SUITE_END() // ConstructorsToDestructors_tests
BOOST_AUTO_TEST_SUITE_END() // Classes
BOOST_AUTO_TEST_SUITE_END() // Cpp
