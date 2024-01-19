import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Replace a string in all JSON and TXT files in a directory and its subdirectories.")
    parser.add_argument("--search", default='/home/jicredt_data/dsj/CDdata/WHUCD/cut_data', help="The string to search for.")
    parser.add_argument("--replace", default='/home/ps/HDD/zhaoyq_data/DATASET/WHUCD', help="The string to replace with.")

    args = parser.parse_args()

    # Check if the specified directory exists
    if not os.path.exists(args.replace):
        print(f"The directory '{args.replace}' does not exist.")
        return

    # Define the file extensions to search for
    file_extensions = ['json', 'txt']

    # Traverse the specified directory and its subdirectories for matching files
# 遍历当前路径及其子文件夹下的所有文件
    for root, _, files in os.walk(args.replace):
        for file in files:
            if file.endswith(tuple(file_extensions)):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 在文件内容中查找并替换字符串
                    modified_content = content.replace(args.search, args.replace)

                    # 将修改后的内容写回文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)

                    print(f'Successfully modified {file_path}')
                except Exception as e:
                    print(f'Error modifying {file_path}: {str(e)}')

if __name__ == "__main__":
    main()
