#!/bin/bash

echo "Height and width of each find and replace group found in lookup.svg:"

# Using the Python script we already ran to get the data
python /workspace/print_dimensions.py | grep "Group" | while read line; do
    echo "$line"
done