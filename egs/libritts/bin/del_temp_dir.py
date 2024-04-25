import os
import shutil

def remove_folders_with_name(root_dir, name_contains):
    for root, dirs, files in os.walk(root_dir, topdown=False):  # topdown=False表示从子文件夹开始遍历
        for dir_name in dirs:
            if name_contains in dir_name:
                # 构建完整的文件夹路径
                folder_path = os.path.join(root, dir_name)
                # 删除文件夹
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")

# 使用示例
root_directory = "/home/v-zhijunjia/data/tts_test_librispeech/LibriSpeech/filtered_4s_10s-test-spk-wer"  # 替换为你要遍历的根目录的路径
name_to_search = "temp"  # 你想要查找并删除的文件夹名包含的字符串
remove_folders_with_name(root_directory, name_to_search)

