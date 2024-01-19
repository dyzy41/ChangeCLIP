# mmsegmentation修改笔记
1. 在configs/_base_/datasets中添加txt.py，设定基础的txt配置文件。
2. 在mmseg/datasets中添加txt.py，在dataset库中添加txt形式的数据接口函数，同时在__init__.py中添加数据接口函数。
3. 在configs/_base_/datasets修改data_root参数，输入正确的train.txt, val.txt, test.txt的路径。
4. 修改目标网络结构配置文件，尝试训练。


# 备注：
1. txt形式的数据读取接口在mmseg/datasets/txt.py代码的242-256行。
2. 如果标签是0/255的，可以在mmseg/datasets/transforms/loading.py的120行进行修改。
3. 如果说想修改读取图片的函数，可以在配置文件里修改这个dict(type='LoadImageFromFile'),然后在mmseg/datasets/transforms/loading.py这个代码里进行修改。


# 网络修改
1. 首先要去看mmseg/models/segmentors文件夹，encoder_decoder.py是完整的分割的模块，里面存放了整个分割功能部分所依赖的函数模块。
2. 如果需要修改网络，可以根据encoder_decoder.py里面去进行修改。