import psutil
import time
import datetime

"""
获取系统基本信息
"""

EXPAND = 1024 * 1024


def mems():
    """ 获取系统内存使用情况 """
    mem = psutil.virtual_memory()
    total_mem = mem.total / EXPAND
    used_mem = mem.used / EXPAND
    idle_mem = mem.total / EXPAND - mem.used / (1024 * 1024)
    # mem_str = " 内存状态如下:\n"
    # mem_str += "   系统的内存容量为: " + str(mem.total / EXPAND) + " MB\n"
    # mem_str += "   系统的内存已使用容量为: " + str(mem.used / EXPAND) + " MB\n"
    # mem_str +=  "   系统可用的内存容量为: " + str(mem.total / EXPAND - mem.used / (1024 * 1024)) + " MB\n"

    return total_mem, used_mem, idle_mem
    # mem_str += "   内存的buffer容量为: " + str(mem.buffers / EXPAND) + " MB\n"
    # mem_str += "   内存的cache容量为:" + str(mem.cached / EXPAND) + " MB\n"


used_meme = []
while True:
    time.sleep(1)
    total_mem, used_mem, idle_mem = mems()
    used_meme.append(used_mem)
    print(f"used mem {used_mem} MB")
