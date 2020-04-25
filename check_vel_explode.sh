#!/bin/bash
echo "checking log file:" $1
vel_exp_num=`cat $1 | grep -i vel | wc -l`
normal_end_num=`cat $1 | grep -i episode | wc -l`
echo "velocity explode number: $vel_exp_num" 
echo "normal end number: $normal_end_num" 
exp_percent=`echo "scale=5;100*$vel_exp_num/$normal_end_num" | bc`

echo "expolode percent:0"$exp_percent%

echo "---------show torque lim number--------"
echo "num  joint"
cat $1 | grep -i lim | awk '{print $4}' | sort -n| uniq -c | sort