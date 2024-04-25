from pydub import AudioSegment  
from pydub.playback import play  
import os
# 加载参考音频文件  

ref_dir = "/home/v-zhijunjia/demo/iclr_final/vctk_iclr/source"
tgt_dir = "/home/v-zhijunjia/demo/iclr_final/vctk_iclr/ours_icml"
source_file_name = "p248_037.wav"

ref_audio = AudioSegment.from_file(os.path.join(ref_dir, source_file_name))  
  
# 加载目标音频文件（您希望调整音量的文件）  
target_audio = AudioSegment.from_file(os.path.join(tgt_dir, source_file_name))
  
# 计算参考音频的平均响度（RMS）  
ref_dBFS = ref_audio.dBFS  
  
# 计算目标音频的平均响度  
target_dBFS = target_audio.dBFS  
  
# 计算两者的响度差  
difference_dBFS = ref_dBFS - target_dBFS  
  
# 调整目标音频的音量，使其与参考音频的音量更接近  
matched_target_audio = target_audio.apply_gain(difference_dBFS)  
  
# 导出调整后的目标音频  
matched_target_audio.export(os.path.join(tgt_dir, "matched_"+source_file_name), format="wav")  
