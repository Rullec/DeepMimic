#!/usr/bin/zsh
server_173='jiafeng@192.168.1.173:/home/jiafeng/playground/DeepMimic/log'
server_121='cad@10.76.5.121:/home/cad/playground/project/DeepMimic/log'

if [ "$1" = "121" ];then
        scp -i ~/.ssh/id_rsa $server_121 ./log_121;
        python plot.py ./log_121
elif [ "$1" = "173" ]; then
        scp -i ~/.ssh/id_rsa $server_173 ./log_173;
        python plot.py ./log_173
else
        exit 0;
fi


