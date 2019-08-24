import sys, os
import json

project_dir = "/home/xudong/Projects/controller-space"

def read_batch_info(project_dir_path = project_dir,\
     model_dir_path_lst = ["data/saved_ppo_models"], env_name_lst = ["NewPendulum-v0"]):
    '''
        read from project_dir_path/model_dir_path/env_name/...
        there are lots of subdirs and each of them is named by their configuration, different models trained in these diversity envs are storaged in it.

        return model path list and configuration path list
    '''
    def get_ret_from_name(name):
        pos = name.find(".ckpt")
        if pos == -1:
            return None, None
        else:
            return float(name[:pos]), name[:pos]

    def get_single_conf(conf_path):
        '''
            从目录中搜索conf, 返回路径
        '''
        if os.path.exists(conf_path) == False:
            raise(FileNotFoundError)
        if os.path.isdir(conf_path) == False:
            raise(FileExistsError)
    
        # get conf file
        files = os.listdir(conf_path)
        conf_file_name = str(os.path.split(conf_path)[-1]) + ".conf"
        for i in files:
            # print((conf_file_name, i))
            if i == conf_file_name:
                conf_file_name = i
        try:
            assert len(conf_file_name) !=0, "conf file doesn't exist!"
        except:
            return None

        # get data dict path
        content_path = os.path.join(conf_path, conf_file_name)
    
        return content_path

    def select_model_from_dir(dir_path):
        max_ret = -sys.float_info.max
        selected_model_name = ""
        for model_name in os.listdir(dir_path):
            # print(model_name)
            num, name = get_ret_from_name(model_name)
            if num == None:
                continue
            if num > max_ret:
                max_ret = num
                selected_model_name = name + ".ckpt"
        return os.path.join(dir_path, selected_model_name)

    # model path
    # model_conf_dict
    model_lst = []
    conf_lst = []
    for model_dir_path in model_dir_path_lst:
        for env_name in env_name_lst:
            # model_env_dir_path = 
            # print(model_env_dir_path)
            env_path = os.path.join(project_dir_path, model_dir_path, env_name)
            for conf_dir_path in os.listdir(env_path):                
                final_dir_path = os.path.join(env_path, conf_dir_path)
                if os.path.isdir(final_dir_path) == False:
                    print("[warn] %s is not dir, skip")
                    continue
                # get model path
                model_path = select_model_from_dir(final_dir_path)
                # read conf path
                conf_path = get_single_conf(final_dir_path)

                # 异常检测
                if model_path != None:
                    model_lst.append(model_path)
                if conf_path != None:
                    conf_lst.append(conf_path)
                # print(select_model_from_dir(model_conf_dir_path))

    assert len(model_lst) == len(conf_lst)
    return model_lst, conf_lst

if __name__ == "__main__":
    print(read_batch_info())