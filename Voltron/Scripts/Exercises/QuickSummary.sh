#!/bin/bash
#
# REV:    0.1.A (Valid are A, B, D, T, Q, and P)
#               (For Alpha, Beta, Dev, Test, QA, and Production)
#
# PLATFORM: Not Platform Dependent (mostly Linux)
#
#
# PURPOSE: Quick summary/look up of commands.
#
# set -n   # Uncomment to check script syntax, without execution.
#          # NOTE: Do not forget to put the # comment back in or
#          #       the shell script will never execute!
# set -x   # Uncomment to debug this shell script
#
##########################################################
#         DEFINE FILES AND VARIABLES HERE
##########################################################

# cf. https://stackoverflow.com/questions/9889938/shell-script-current-directory
# Shell script currenty directory?

# This works. Expands variables and prints a little + sign before the line.
# cf. https://stackoverflow.com/questions/2853803/how-to-echo-shell-commands-as-they-are-executed
# set -o xtrace

echo "Script executed from: ${PWD}"
echo "Script executed from: $PWD"

BASEDIR=$( dirname $0 )
echo "Script location: ${BASEDIR}"
echo "Script location: $BASEDIR"

# cf. Mastering Unix Shell Scripting, 2nd ed. Randal Michael
# pp. 863, Using the dirname and basename Commands
# When we need to separate a filename and directory path, use dirname and
# basename.
# dirname will return directory part of full-path filename.
echo "This is \$0: $0"
DIR="( cd "$( dirname "$0" )" && pwd )"
echo "This is DIR: $DIR"
# Then use $DIR as "$DIR/path/to/file"

##########################################################
#              DEFINE FUNCTIONS HERE
# cf. Mastering Unix Shell Scripting, 2nd. Ed. Randal Michael. pp. 4-5.
# Functions. A function has the following form:
# function function_name
# {
#   commands to execute
# }
# or
# function_name ()
# {
#   commands to execute
# }
#
# cf. https://stackoverflow.com/questions/3236871/how-to-return-a-string-value-from-a-bash-function
# shell script function return string
# tl;dr you can't. Manipulate an input argument.
##########################################################


##########################################################
#               BEGINNING OF MAIN
##########################################################


# End of script

