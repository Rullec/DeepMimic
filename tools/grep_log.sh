
# log_filename="../log_8_worker" # ../log_4_worker"
log_filename="../nohup.out" # ../log_4_worker"
test_keyword="Test"
train_keyword="Train"

for i in $log_filename;
do
	# file detect
	#echo $i
	if [ -e $i ];
	then 	
		echo "$i found"
	else
		echo "$i not found"
	fi

	# grep num then output
	test_num=`cat $i |grep $test_keyword| awk '{print $4}'`
	train_num=`cat $i |grep $train_keyword| awk '{print $4}'`
	output_filename_test=$i'.number.test'
	output_filename_train=$i'.number.train'
	
	echo $output_filename_test" output"
	echo $output_filename_train" output"
	
	echo $test_num >$output_filename_test
	echo $train_num >$output_filename_train
done
