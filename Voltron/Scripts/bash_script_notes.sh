#!/bin/bash

# This is normally first line of script, system still uses this line, instructs interpreter of system to execute script.
# If we don't add this line, then commands will be run within current shell; it may cause issues if we run another shell.
#!/bin/bash

# cf. Mastering Linux Shell Scripting

# Do chmod +x bash_script_notes.sh too

# $# is number of arguments. -gt is greater than.
if [ $# -gt 0 ]; then
  echo "$# number of arguments, Hello $1"
else
  echo "${#} number of arguments"
fi