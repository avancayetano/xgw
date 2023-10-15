#!/usr/bin/bash

# Sample calls:
# ./run_mcl.sh ../data/weighted/ ./ 4
# ./run_mcl.sh ../data/weighted/ all_edges/ 4
# ./run_mcl.sh ../data/weighted/ all_edges/cross_val/ 4
# ./run_mcl.sh ../data/weighted/ all_edges/features/ 4
# ./run_mcl.sh ../data/weighted/ 20k_edges/ 4

# NOTE: running ./run_mcl.sh ../data/weighted/ ./ 4 will run MCL (with inflation=4)
# on all the files inside ../data/weighted/

# Likewise, running ./run_mcl.sh ../data/weighted/ all_edges/cross_val/ 4 will run
# MCL on all the files inside ../data/weighted/all_edges/cross_val/

parent_dir=$1
edges_dir=$2
inflation=$3

input_dir="$parent_dir$edges_dir"
output_dir="../data/clusters/"

TEMP="$IFS"
IFS=$'\n'
for file in $(find "$input_dir" -iregex ".*\.\(csv\|txt\)$"); do
    odir=$(echo "$file" | perl -nle 'm/(.+)\/.+\.((txt)|(csv))$/; print $1')
    odir=$output_dir${odir#"$parent_dir"}
    $(~/local/bin/mcl "$file" --abc -I "$inflation" -odir "$odir")
    echo "--------------------- DONE CLUSTERING ------------------------------"
done
IFS="$TEMP"
