import librosa  
import librosa.display  
import numpy as np  
import matplotlib.pyplot as plt  
import textwrap  
import matplotlib.patches as mpatches  
import matplotlib.lines as mlines  
  
def create_pitch_contour_plot(file_path, accent_name, time_labels, output_file_name):  
    audio_data, sample_rate = librosa.load(file_path)  
    pitch_values, voiced_flag, voiced_probs = librosa.pyin(audio_data,  
                                                           fmin=librosa.note_to_hz('C2'),  
                                                           fmax=librosa.note_to_hz('C7'))  
  
    pitch_values_semitones = librosa.hz_to_midi(pitch_values)  
    times = librosa.times_like(pitch_values)  
  
    plt.figure(figsize=(12, 4))  
    plt.plot(times, pitch_values_semitones, 'o-', markersize=3)  
    plt.ylabel('Pitch (semitones)',  fontsize=20)  
    plt.xlabel('Time (s)',  fontsize=20)  
    plt.title(f'{accent_name} accent', fontsize=20)  
  
    if np.isfinite(pitch_values_semitones).any():  
        plt.ylim([np.nanmin(pitch_values_semitones) - 2, np.nanmax(pitch_values_semitones) + 8])  
  
    # Add labels and trend lines to the plot  
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'plum']  # Add or modify colors in this list  
    color_idx = 0
    for label_start_time, label_end_time, label_text, given_pitch, trend_color in time_labels:  
        start_idx = np.searchsorted(times, label_start_time)  
        end_idx = np.searchsorted(times, label_end_time)  
        if start_idx >= 0 and start_idx < len(pitch_values_semitones):  
            pitch_segment = pitch_values_semitones[start_idx:end_idx]  
            if given_pitch is not None:  
                pitch_label_value = given_pitch  
            else:  
                pitch_label_value = np.median(pitch_segment[~np.isnan(pitch_segment)])  
            wrapped_label_text = '\n'.join(textwrap.wrap(label_text, width=10))  
            plt.annotate(wrapped_label_text, xy=(label_start_time, pitch_label_value),  
                         xytext=((label_start_time + label_end_time) / 2, pitch_label_value),  
                         fontsize=20, ha='center', va='bottom', multialignment='center')  # adjust lable
  
            line_color = trend_color  
            if trend_color == 'red':  
                end_height = pitch_label_value + 4  
            elif trend_color == 'black':  
                end_height = pitch_label_value - 4  
            else:  # brown  
                end_height = pitch_label_value + 2  
  
            arrow_start_time = label_start_time + (label_end_time - label_start_time) * 1 / 3  
            arrow_end_time = (label_start_time + label_end_time) / 2  
            arrow = mpatches.FancyArrowPatch((arrow_start_time, pitch_label_value + 2 if line_color != 'black' else pitch_label_value - 2),  
                                             (arrow_end_time, end_height),  
                                             arrowstyle='-|>', mutation_scale=10,  
                                             lw=2, color=line_color,  
                                             connectionstyle="arc3,rad=-.5" if line_color == 'black' else "arc3,rad=.5", alpha=1)  
            plt.gca().add_patch(arrow)  
  
            line_color = colors[color_idx % len(colors)]  # Use the next color in the list  
            plt.axvline(label_start_time, color=line_color, linestyle='--', linewidth=1)    
            plt.axvline(label_end_time, color=line_color, linestyle='--', linewidth=1)    
          
        color_idx += 1 
  
    # Set x-axis limits  
    duration = librosa.get_duration(audio_data, sample_rate)  
    plt.xlim(0, duration)  
  
    # Add legend  
    red_arrow = mlines.Line2D([], [], color='red', marker='>',  
                               linestyle='-', linewidth=2, markersize=10,label='Pitch up')
    black_arrow = mlines.Line2D([], [], color='black', marker='>',
    linestyle='-', linewidth=2, markersize=10, label='Pitch down')
    brown_arrow = mlines.Line2D([], [], color='brown', marker='>',
    linestyle='-', linewidth=2, markersize=10, label='Pitch steady')
    plt.legend(handles=[red_arrow, black_arrow, brown_arrow], loc='upper right', fontsize=20)

    # Save the plot  
    plt.savefig(f'{output_file_name}', dpi=300, bbox_inches='tight', pad_inches=0.5)  

    # Show the plot  
    plt.show()  



# Time labels as a list of tuples [(start_time, end_time, text), ...]  
time_labels = [(0, 0.15, 'It\'s', None, 'red'), (0.21, 0.4, 'also', None, 'red'), (0.52, 0.87, 'very', None, 'red'), (0.88, 1.34, 'valuable', None, 'black')]  
  
# Replace with the path to your Indian English audio file  
indian_accent_file_path = '/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk/source_ac/source_p248_260.wav'  
create_pitch_contour_plot(indian_accent_file_path, 'general American-English', time_labels, 'pitch_contour_general_american_v3.png')  

# time_labels = [(0, 0.21, 'It\'s',58, 'red'), (0.21, 0.55, 'also', None, 'black'), (0.6, 0.83, 'very', None, 'brown'), (0.87, 1.33, 'valuable', None, 'black')]  
# # Replace with the path to your American English audio file  
# american_accent_file_path = '/home/v-zhijunjia/data/plots_pictures/trimmed_audio_p248_260.wav'  
# create_pitch_contour_plot(american_accent_file_path, 'Native English', time_labels)  

time_labels = [(0, 0.14, 'It\'s',None, 'red'), (0.16, 0.41, 'also', None, 'black'), (0.43, 0.84, 'very', None, 'brown'), (0.86, 1.3, 'valuable', None, 'black')]  
# Replace with the path to your American English audio file  
american_accent_file_path = '/home/v-zhijunjia/data/accent_iclr/iclr_final/ac_vctk/source_ac/ac_baseline_20cases_p248_260.wav'  
create_pitch_contour_plot(american_accent_file_path, 'Indian-English', time_labels, 'pitch_contour_Indian_English_v3.png')  
