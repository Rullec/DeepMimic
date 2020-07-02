#!/usr/bin/zsh
scp -i ~/.ssh/id_rsa cad@10.76.5.121:/home/cad/playground/project/DeepMimic/log .
python plot.py ./log
echo '' >> ./log