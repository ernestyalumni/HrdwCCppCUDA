#/bin/sh

# Exercises: 2.1. Write a shell script sum.sh that takes an unspecified number
# of command line arguments (up to 9) of ints and finds their sum. Modify the
# code to add a number to the sum only if the number is greater than 10.

sum1=0
for var in "$@"
do
  sum1=`expr $sum1 + $var`
done
printf "sum1 : %s\n" $sum1

sum2=0
for var in "$@"
do
  if [ $var -gt 10 ]
    then
      sum2=`expr $sum2 + $var`
    fi
done
printf "sum2: %s\n" $sum2

# Run like this: sh 2.1sh 2 4 5 -- run the script as 