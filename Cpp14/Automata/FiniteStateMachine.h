//------------------------------------------------------------------------------
/// \file FiniteStateMachine.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  2-Tuple.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or
/// math, sciences, etc.), so I am committed to keeping all my material
/// open-source and free, whether or not sufficiently crowdfunded, under the
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++17 Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#ifndef _AUTOMATA_FINITE_STATE_MACHINE_H_
#define _AUTOMATA_FINITE_STATE_MACHINE_H_

namespace Automata
{

//------------------------------------------------------------------------------
/// \class Automaton
/// \brief Automaton interface
//------------------------------------------------------------------------------
class Automaton
{

  public:

    // Data is gone; ctors gone since there's no data to initialize.

    //--------------------------------------------------------------------------
    /// \fn start
    /// \brief Current state of the FSM is set to the initial state.
    //--------------------------------------------------------------------------
    virtual void start() = 0; // pure virtual function

    //--------------------------------------------------------------------------
    /// \fn transition
    /// \brief Change from current state to another in response to some external
    /// inputs.
    //--------------------------------------------------------------------------
    virtual void transition() = 0; // pure virtual function

    // Add virtual destructor to ensure proper cleanup of data that'll be
    // defined in derived class
    virtual ~Automaton()
    {}

}; // class Automaton

//------------------------------------------------------------------------------
/// \class FiniteStateMachine
/// \brief A finite state machine (FSM) interface
//------------------------------------------------------------------------------
class FiniteStateMachine : public Automaton, public States
{
  public:

    virtual void set_initial_state(States& state) = 0;



}; // class FiniteStateMachine


} // namespace Automata

#endif // _AUTOMATA_FINITE_STATE_MACHINE_H_
