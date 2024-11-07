#!/bin/sh
cmd_list=$(find . \( -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cxx" \))

for files in ${cmd_list}; do
    echo "$files"
    ../exaNBody/scripts/code-format.sh "$files"
done



