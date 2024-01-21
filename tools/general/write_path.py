# import os
#
# dataset_path = r"F:\0Fcode\code220303_seg\whub"  # 将此处替换为您的数据集路径
# output_path = r"D:\0Ecode\code230710_mm\mmseg\mmsegmentation\datasets\whub"  # 将此处替换为您想要保存txt文件的输出路径
#
# # 创建保存txt文件的文件夹
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
#
# # 分别处理train、val和test文件夹
# for folder_name in ["train", "val", "test"]:
#     txt_path = os.path.join(output_path, f"{folder_name}.txt")
#
#     with open(txt_path, "w") as f:
#         image_folder = os.path.join(dataset_path, folder_name, "image")
#         label_folder = os.path.join(dataset_path, folder_name, "label")
#
#         # 获取图片和标签文件夹中的文件列表
#         image_files = sorted(os.listdir(image_folder))
#         label_files = sorted(os.listdir(label_folder))
#
#         # 检查图片和标签文件数量是否匹配
#         if len(image_files) != len(label_files):
#             raise ValueError(f"图片和标签文件数量不匹配：{folder_name}")
#
#         # 写入图片和标签路径到txt文件
#         for image_file, label_file in zip(image_files, label_files):
#             image_path = os.path.join(image_folder, image_file)
#             label_path = os.path.join(label_folder, label_file)
#             f.write(f"{image_path}  {label_path}\n")
#
# print("所有txt文件生成完成！")

######################################################################################################################
import os

dataset_path = "/home/ps/HDD/zhaoyq_data/CDdata/LEVIR-CD"  # 将此处替换为您的数据集路径
output_path = "/home/ps/HDD/zhaoyq_data/CDdata/LEVIR-CD"  # 将此处替换为您想要保存txt文件的输出路径

# 创建保存txt文件的文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 分别处理train、val和test文件夹
for folder_name in ["test"]:
    txt_path = os.path.join(output_path, f"{folder_name}.txt")

    with open(txt_path, "w") as f:
        image_folderA = os.path.join(dataset_path, folder_name, "A")
        image_folderB =  os.path.join(dataset_path, folder_name, "B")
        label_folder = os.path.join(dataset_path, folder_name, "label")

        # 获取图片和标签文件夹中的文件列表
        image_filesA = sorted(os.listdir(image_folderA))
        image_filesB = sorted(os.listdir(image_folderB))
        label_files = sorted(os.listdir(label_folder))

        # 检查图片和标签文件数量是否匹配
        if len(image_filesA) != len(label_files):
            raise ValueError(f"图片和标签文件数量不匹配：{folder_name}")

        # 写入图片和标签路径到txt文件
        for image_fileA, image_fileB, label_file in zip(image_filesA, image_filesB, label_files):
            image_pathA = os.path.join(image_folderA, image_fileA)
            image_pathB = os.path.join(image_folderB, image_fileB)
            label_path = os.path.join(label_folder, label_file)
            f.write(f"{image_pathA}  {image_pathB}  {label_path}\n")

print("所有txt文件生成完成！")
