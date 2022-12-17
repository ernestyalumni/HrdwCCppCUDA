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