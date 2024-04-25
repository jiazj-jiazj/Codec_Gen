import librosa
import soundfile as sf
import os

def trim_audio(input_file, output_file, start_time, end_time):
    audio, sample_rate = librosa.load(input_file, sr=None)  # 保留原始采样率
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    trimmed_audio = audio[start_sample:end_sample]
    sf.write(output_file, trimmed_audio, sample_rate)

def process_folder(input_folder, output_folder, start_time, end_time):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        # 构建完整的文件路径
        input_file = os.path.join(input_folder, file_name)

        # 检查是否为音频文件
        if os.path.isfile(input_file) and file_name.endswith(('.wav', '.flac')):
            output_file = os.path.join(output_folder, file_name)
            trim_audio(input_file, output_file, start_time, end_time)

if __name__ == "__main__":
    input_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/origin_prompt_listen_filter_p_0_6_test_LibrSpeech_6s_prompt"  # 替换为您的输入文件夹路径
    output_folder = "/home/v-zhijunjia/data/data_update/benchmark_librispeech_10speakers/origin_prompt_listen_filter_p_0_6_test_LibrSpeech_6s_prompt_trimmed"  # 替换为您的输出文件夹路径
    start_time = 0  # 裁剪开始时间（秒）
    end_time = 3  # 裁剪结束时间（秒）

    process_folder(input_folder, output_folder, start_time, end_time)
