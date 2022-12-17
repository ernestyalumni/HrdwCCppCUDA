----------------------------- MODULE wire_wayne -----------------------------

\* Everything above the dashes and below the equal signs are ignored.

\* To compile PlusCal, go to File > Translate PlusCal Algorithm
\* To create the model, TLC Model Checker > New Model, or Open Model

\* TLA+ keyword for import is EXTENDS.
EXTENDS Integers

\* Algorithm is inside a comment. Put in comment for translation into PlusCal

\* Whitespace is not significant inside here.

(*--algorithm wire
    variables
        \* We're tracking 2 things in the system state: set of people with accounts, and how much
        \* how much money each of them has.
        \* people is a *set*
        people = {"alice", "bob"},
        \* acc is a function or mapping. For each value in a given set, maps to some output value.
        acc = [p \in people |-> 5],

\* invariant, for all p in the set of people, their account must be greater than or equal to 0.
define
    NoOverdrafts == \A p \in people: acc[p] >= 0
    \* <>[] is the "eventually-always" operator, means that no matter what the algorithm does, in
    \* the end the given equation must eventually be true.
    EventuallyConsistent == <>[](acc["alive"] + acc["bob"] = 10)
end define;

process Wire \in 1..2
    variables

        sender = "alice",
        receiver = "bob",
        amount \in 1..acc[sender];

begin
    \* CheckFunds:
        \* if amount <= acc[sender] then
            \* Withdraw and Deposit are labels; they signify that everything inside them happens in
            \* the same moment in time.
            \* Withdraw:
            \*    acc[sender] := acc[sender] - amount;
            \* Deposit:
            \*    acc[receiver] := acc[receiver] + amount;
        \* end if;
    CheckAndWithdraw:
        if amount <= acc[sender] then
                acc[sender] := acc[sender] - amount;
            Deposit:
                acc[receiver] := acc[receiver] + amount;
        end if;
end process;

end algorithm;*)
\* BEGIN TRANSLATION (chksum(pcal) = "b4e207d" /\ chksum(tla) = "e84f5c17")
VARIABLES people, acc, pc

(* define statement *)
NoOverdrafts == \A p \in people: acc[p] >= 0


EventuallyConsistent == <>[](acc["alive"] + acc["bob"] = 10)

VARIABLES sender, receiver, amount

vars == << people, acc, pc, sender, receiver, amount >>

ProcSet == (1..2)

Init == (* Global variables *)
        /\ people = {"alice", "bob"}
        /\ acc = [p \in people |-> 5]
        (* Process Wire *)
        /\ sender = [self \in 1..2 |-> "alice"]
        /\ receiver = [self \in 1..2 |-> "bob"]
        /\ amount \in [1..2 -> 1..acc[sender[CHOOSE self \in  1..2 : TRUE]]]
        /\ pc = [self \in ProcSet |-> "CheckAndWithdraw"]

CheckAndWithdraw(self) == /\ pc[self] = "CheckAndWithdraw"
                          /\ IF amount[self] <= acc[sender[self]]
                                THEN /\ acc' = [acc EXCEPT ![sender[self]] = acc[sender[self]] - amount[self]]
                                     /\ pc' = [pc EXCEPT ![self] = "Deposit"]
                                ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                                     /\ acc' = acc
                          /\ UNCHANGED << people, sender, receiver, amount >>

Deposit(self) == /\ pc[self] = "Deposit"
                 /\ acc' = [acc EXCEPT ![receiver[self]] = acc[receiver[self]] + amount[self]]
                 /\ pc' = [pc EXCEPT ![self] = "Done"]
                 /\ UNCHANGED << people, sender, receiver, amount >>

Wire(self) == CheckAndWithdraw(self) \/ Deposit(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == (\E self \in 1..2: Wire(self))
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 


=============================================================================
\* Modification History
\* Last modified Fri Dec 16 17:38:17 PST 2022 by topolo
\* Created Fri Jun 18 12:10:04 PDT 2021 by topolo
