#!/bin/sh
#
# REV:    1.1.A (Valid are A, B, D, T, Q, and P)
#               (For Alpha, Beta, Dev, Test, QA, and Production)
#
# PLATFORM: Not platform dependent
#
#
# PURPOSE: This script is used to process all of the tokens which are pointed to
# by the command-line arguments, $1, $2, $3, etc.
#
# REV LIST:
#
# set -n   # Uncomment to check script syntax, without execution.
#          # NOTE: Do not forget to put the # comment back in or
#          #       the shell script will never execute!
# set -x   # Uncomment to debug this shell script
#
# cf. Mastering Unix Shell Scripting, 2nd Ed Randal Michael

# cf. pp. 19, Using the echo Command Correctly. In Bash shell, we must add the
# -e switch to echo command, echo -e "\n" for 1 new line.

# Set up the correct echo command usage. Many Linux distributions will execute
# in Bash even if the script specifies Korn shell. Bash shell requires we use
# echo -e when we use \n, \c, etc.
case $SHELL in
  */bin/Bash) alias echo="echo -e"
    ;;
esac

##########################################################
#         DEFINE FILES AND VARIABLES HERE
##########################################################

# Initialize all variables

total=0 # Initialize the total counter to zero

##########################################################
#              DEFINE FUNCTIONS HERE
##########################################################
 

##########################################################
#               BEGINNING OF MAIN
##########################################################

# Start a while loop

#while true
#do
#  total=`expr $total + 1` # A little math in the shell script, a running total
    # of tokens processed.
#  token=$1 $ We always point to the $1 argument with a shift
    # process each $token
#  printf "token: %s" $token
#  echo $token
#  shift
#done

for token in $*
do
  total=`expr $total + 1`
  printf "token: %s" $token
  echo $token
done

echo "Total number of tokens processed: $total"

# cf. pp. 25 Setting Traps.
# When a program is terminated before it would normally end, we can catch an
# exit signal. This is called a trap.


# End of script

