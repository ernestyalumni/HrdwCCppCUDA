#!/bin/sh

# cf. Lecture 02 - Shell Scripting, 15-123S09 CS CMU

echo $1

ls $1

# Finds current date, and uses cut (string tokenizer) to extract a specific part
# of the date. Then it assigns that to variable day.
# The day is then used by printf statement.
day=`date | cut -d" " -f1`
printf "Today is %s.\n" $day

