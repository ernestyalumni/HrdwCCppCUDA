# TLA+ guide and notes

https://github.com/Apress/practical-tla-plus

## Running on Fedora Linux

Go to the, change ('cd') into the, subdirectory `toolbox` of the "binary", "executable", "program" and run the binary `./toolbox`.

## Compiling, Checking

### Compiling PlusCal

To compile PlusCal, go to File > Translate PlusCal Algorithm

## Errors

### Reading Errors

cf. pp. 18, Ch. 1, An Example, Practical TLA

`<Initial PRedicate>` - starting state we picked; all variables have an assignment here,
`pc` - current label each process is on
`amount` is corresponding wire amount for each process

## Temporal Properties

cf. pp. 19, Ch. 1, An Example, Practical TLA+

**Temporal Property** - check that every possible "lifetime" of the algorithm, from start to finish, objects something that relates different states in the sequence to each other.

## MBT's `Hello World` example for a State Machine in TLA+

The invariants in this example are `NothingUnexpectedInNetwork` and `NotBobIsHappy`. You can "check" for these invariants, for each one at a time or both.

To do so, you either change the `.cfg` manually, or in the TLA+ Toolbox GUI, you make or open a "Model" for the TLA+ specification, go to Model Overview -> What to check? -> Invariants. If you don't see them, go to the right and click "Add" and make sure to type the name exactly. Then click in the center for the check box.
