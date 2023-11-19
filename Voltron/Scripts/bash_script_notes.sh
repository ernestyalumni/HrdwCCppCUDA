#!/bin/bash

# This is normally first line of script, system still uses this line, instructs interpreter of system to execute script.
# If we don't add this line, then commands will be run within current shell; it may cause issues if we run another shell.
#!/bin/bash

# cf. Mastering Linux Shell Scripting

# Do chmod +x bash_script_notes.sh too

get_number_of_arguments()
{
  # $# is number of arguments. -gt is greater than.
  if [ $# -gt 0 ]; then
    echo "$# number of arguments, Hello $1"
  else
    echo "${#} number of arguments"
  fi
}

check_directory_contains_name()
{
  # Get current directory path.
  current_dir="$PWD"
  echo "Current directory: '$PWD'"

  # Specify directory name you want to check for
  target_dir="HrdwCCppCUDA"

  # Check if current directory ends with target directory name.
  if [[ "$current_dir" == *"/$target_dir" ]]; then
    echo "Current directory ends with '$target_dir'. Performing some operations."
    return 0
  else
    echo "Current directory does not end with '$target_dir'. Skipping operations."
    return 1
  fi
}

# cf. Lecture 02 - Shell Scripting, 15-123S09 CS CMU

# Quotes

# unquoted strings are normally interpreted
# "quoted strings are basically literals -- but $variables are evaluated"
# 'quoted strings are absolutely literally interpreted'
# `commands in quotes like this are executed, their output is then inserted as
# if it were assigned to a variable and then that variable was evaluated`

# Expressions, Strings, pp. 92, Ch. 6 Expressions of
# Linux Shell Scripting with Bash, Ken. O. Burtch.
#
# -z s--(zero length) True if string is empty
# -n s (or just s) -- (not null) True if string is not empty
# s1 = s2 - True if string s1 is equal to s2
# s1 != S2
# s1 < s2
# s1 >> s2

# pp. 100, Ch. 6 Expressions
# Arithmetic Tests
# n1 -eq n2
# n1 -ne n2
# n1 -lt n2

# Call function.
# https://superuser.com/questions/694501/what-does-mean-as-a-bash-script-function-parameter
# $@ variable expands to all the parameters used when call the function. $@ has
# each parameter as a separate quoted string, whereas $* has all parameters as a
# single string.
get_number_of_arguments "$@"

check_directory_contains_name