#!/bin/sh # This line should always be the first line in your script
# A simple script
whoami
date
pwd

# cf. Lecture 02 - Shell Scripting, 15-123S09 CS CMU
# #!/bin/sh tells shell to invoke /bin/sh to run script. This is necessary
# because different users might be using different shells.

# To create new variables, simply assign them a value:

value="dir"
echo $value

# I/O Commands
# echo - To print to stdout
# read - To obtain values from stdin

# Command Line Arguments
# $# - represents total number of arguments (much like argv) - except command
# $0 - represents name of script, as invoked
# $1, $2, $3, ... $8, $9 - first 9 command line arguments
# $* - all command line arguments OR
# $@ - all command line arguments

echo $#
echo $0
echo $*
echo $@