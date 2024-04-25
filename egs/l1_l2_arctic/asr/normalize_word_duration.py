word_offsets = [{'word': 'WE', 'start_time': 0.0, 'end_time': 0.06}, {'word': 'ALSO', 'start_time': 0.22, 'end_time': 0.5}, {'word': 'NEED', 'start_time': 0.64, 'end_time': 0.82}, {'word': 'A', 'start_time': 0.94, 'end_time': 0.96}, {'word': 'SMALL', 'start_time': 1.06, 'end_time': 1.36}, {'word': 'PLASTIC', 'start_time': 1.44, 'end_time': 1.96}, {'word': 'SNAKEAND', 'start_time': 2.08, 'end_time': 2.76}, {'word': 'A', 'start_time': 2.86, 'end_time': 2.88}, {'word': 'BIG', 'start_time': 2.98, 'end_time': 3.18}, {'word': 'TIE', 'start_time': 3.28, 'end_time': 3.56}, {'word': 'FRODG', 'start_time': 3.68, 'end_time': 4.02}, {'word': 'FOR', 'start_time': 4.16, 'end_time': 4.28}, {'word': 'THE', 'start_time': 4.38, 'end_time': 4.46}, {'word': 'KIDS', 'start_time': 4.52, 'end_time': 4.82}]
transcript1 = [  
    {'start_time': d['start_time'], 'end_time': d['end_time'], 'text': d['word'].capitalize().lower()}  
    for d in word_offsets  
]  
  
# 请注意，您提供的时间偏移值在新列表`transcript1`中似乎有一些错误（例如，结束时间小于开始时间）。  
# 因此，我将使用原始列表`word_offsets`中的时间偏移。  
# 如果您有正确的时间偏移，请将它们替换到`word_offsets`中。  
  
print(transcript1)  
