-------------------------------------- MODULE SortingMachine --------------------------------------

EXTENDS FiniteSets, Integers, Sequences, TLC 

(*--algorithm SortingMachine
variables
  capacity = [trash |-> 10, recycle |-> 10],
  bins = [trash |-> {}, recycle |-> {}],
  count = [trash |-> 0, recycle |-> 0],
  items = <<
    [type |-> "recycle", size |-> 5],
    [type |-> "trash", size |-> 5],
    [type |-> "recycle", size |-> 4],
    [type |-> "recycle", size |-> 3]
  >>,
  curr = ""; \* helper: current item

begin
  while items /= <<>> do
    curr := Head(items);
    items := Tail(items);
    if curr.type = "recycle" /\ curr.size < capacity.recycle then
      bins.recycle := bins.recycle \union {curr};
      capacity.recycle := capacity.recycle - curr.size;
      count.recycle := count.recycle + 1;
    elsif curr.size < capacity.trash then
      bins.trash := bins.trash \union {curr};
      capacity.trash := capacity.trash - curr.size;
      count.trash := count.trash + 1;
    end if;
  end while;

  assert capacity.trash >= 0 /\ capacity.recycle >= 0;
  assert Cardinality(bins.trash) = count.trash;
  assert Cardinality(bins.recycle) = count.recycle;

end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "c5de0e3a" /\ chksum(tla) = "9ba45585")
VARIABLES capacity, bins, count, items, curr, pc

vars == << capacity, bins, count, items, curr, pc >>

Init == (* Global variables *)
        /\ capacity = [trash |-> 10, recycle |-> 10]
        /\ bins = [trash |-> {}, recycle |-> {}]
        /\ count = [trash |-> 0, recycle |-> 0]
        /\ items =         <<
                     [type |-> "recycle", size |-> 5],
                     [type |-> "trash", size |-> 5],
                     [type |-> "recycle", size |-> 4],
                     [type |-> "recycle", size |-> 3]
                   >>
        /\ curr = ""
        /\ pc = "Lbl_1"

Lbl_1 == /\ pc = "Lbl_1"
         /\ IF items /= <<>>
               THEN /\ curr' = Head(items)
                    /\ items' = Tail(items)
                    /\ IF curr'.type = "recycle" /\ curr'.size < capacity.recycle
                          THEN /\ bins' = [bins EXCEPT !.recycle = bins.recycle \union {curr'}]
                               /\ capacity' = [capacity EXCEPT !.recycle = capacity.recycle - curr'.size]
                               /\ count' = [count EXCEPT !.recycle = count.recycle + 1]
                          ELSE /\ IF curr'.size < capacity.trash
                                     THEN /\ bins' = [bins EXCEPT !.trash = bins.trash \union {curr'}]
                                          /\ capacity' = [capacity EXCEPT !.trash = capacity.trash - curr'.size]
                                          /\ count' = [count EXCEPT !.trash = count.trash + 1]
                                     ELSE /\ TRUE
                                          /\ UNCHANGED << capacity, bins, 
                                                          count >>
                    /\ pc' = "Lbl_1"
               ELSE /\ Assert(capacity.trash >= 0 /\ capacity.recycle >= 0, 
                              "Failure of assertion at line 33, column 3.")
                    /\ Assert(Cardinality(bins.trash) = count.trash, 
                              "Failure of assertion at line 34, column 3.")
                    /\ Assert(Cardinality(bins.recycle) = count.recycle, 
                              "Failure of assertion at line 35, column 3.")
                    /\ pc' = "Done"
                    /\ UNCHANGED << capacity, bins, count, items, curr >>

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == pc = "Done" /\ UNCHANGED vars

Next == Lbl_1
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(pc = "Done")

\* END TRANSLATION 

===================================================================================================
