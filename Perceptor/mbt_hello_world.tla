\* cf. https://mbt.informal.systems/docs/tla_basics_tutorials/hello_world.html
\* https://github.com/informalsystems/modelator/blob/main/jekyll/docs/tla_basics_tutorials/models/hello_world/hello_world.tla
\* 'Hello world' using TLC

-------------------------- MODULE mbt_hello_world --------------------------

\* extend (import) modules from the standard library
\* import Sequences module from the standard library.
\* It provides the Sequence data structure. It is a list.
EXTENDS Sequences

\* The state machine.
VARIABLES
    alices_outbox,
    network,
    bobs_mood,
    bobs_inbox
    
Init ==
    /\ alices_outbox = {} \* Alice has sent nothing (empty set)
    /\ network = {} \* AND so is the network
    /\ bobs_mood = "neutral" \* AND Bob's mood is neutral
    /\ bobs_inbox = <<>> \* AND Bob's inbox is an empty Sequence (list)

AliceSend(m) ==
    /\ m \notin alices_outbox
    /\ alices_outbox' = alices_outbox \union {m}
    /\ network' = network \union {m}
    /\ UNCHANGED <<bobs_mood, bobs_inbox>>

NetworkLoss ==
    /\ \E e \in network: network' = network \ {e}
    /\ UNCHANGED <<bobs_mood, bobs_inbox, alices_outbox>>
    
NetworkDeliver ==
    /\ \E e \in network:
        /\ bobs_inbox' = bobs_inbox \o <<e>>
        /\ network' = network \ {e}
    /\ UNCHANGED <<bobs_mood, alices_outbox>>

BobCheckInbox ==
    /\ bobs_mood' = IF bobs_inbox = <<"hello", "world">> THEN "happy" ELSE "neutral"
    /\ UNCHANGED <<network, bobs_inbox, alices_outbox>>
    
Next ==
    \/ AliceSend("hello")
    \/ AliceSend("world")
    \/ NetworkLoss
    \/ NetworkDeliver
    \/ BobCheckInbox
    
NothingUnexpectedInNetwork == \A e \in network: e \in alices_outbox

NotBobIsHappy ==
    LET BobIsHappy == bobs_mood = "happy"
    IN ~BobIsHappy

=============================================================================
\* Modification History
\* Last modified Fri Dec 23 21:37:37 PST 2022 by topolo
\* Created Fri Dec 23 21:20:00 PST 2022 by topolo
