#!/bin/sh

# cf. Lecture 02 - Shell Scripting, 15-123S09 CS CMU

# expr can be used to manipulate variables, normally interpreted as strings, as
# integers.

sum=`expr $1 + $2`
printf "%s + %s = %s\n" $1 $2 $sum

printf "The shell's pid: %s" $$
echo $$
printf "The exit status of last program to exit: $s" $?
echo $?

test "$LOGNAME"=guna
echo $?

# You can think of the [] operator as a form of the test command. But, one very
# important note -- there must be a space to the inside of each of the brackets.
if [ "$LOGNAME"="guna" ]
then
  printf "%s is logged in" $LOGNAME
else
  printf "Intruder! Intruder!"
fi

