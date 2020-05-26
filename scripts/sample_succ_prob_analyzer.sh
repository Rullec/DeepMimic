echo "checking log file:" $1
fail_num=`cat $1 | grep -i "CheckTerminate" | wc -l`
succ_num=`cat $1 | grep -i "Timer" | wc -l`
total_num=`echo "$fail_num+$succ_num" | bc`
echo "succ_num = " $succ_num, `echo "scale=3;$succ_num/$total_num*100" | bc`"%"
echo "fail_num = " $fail_num, `echo "scale=3;$fail_num/$total_num*100" | bc`"%"
echo "total_num = " $total_num