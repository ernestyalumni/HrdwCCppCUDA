#/bin/sh

# cf. Lecture 02 - Shell Scripting, 15-123S09 CS CMU

# 2.2 Write a shell script that takes the name of a path and counts all the
# subdirectories (recursively)

# Run the script as:
# % sh 2.2.sh /afs/andrew/course/15/123/handin

echo $1

ls -R $1 | wc -l