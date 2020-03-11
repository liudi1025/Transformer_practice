import json

# read a large .json file: https://zhuanlan.zhihu.com/p/57533731
# 分行读
with open('/data/chejin/data/aidata.json') as f:
    for i in range(2):
        line = f.readline()
        print(line)
        print(type(line))
        #print(type(json.load(line)))
