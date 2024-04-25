"""
This recipe supports two corpora: LibriTTS and LibriTTS-R.

---

LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members. The LibriTTS corpus is designed for TTS research. It is derived from the original materials (mp3 audio files from LibriVox and text files from Project Gutenberg) of the LibriSpeech corpus. The main differences from the LibriSpeech corpus are listed below:
The audio files are at 24kHz sampling rate.
The speech is split at sentence breaks.
Both original and normalized texts are included.
Contextual information (e.g., neighbouring sentences) can be extracted.
Utterances with significant background noise are excluded.
For more information, refer to the paper "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech", Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu, arXiv, 2019. If you use the LibriTTS corpus in your work, please cite this paper where it was introduced.

---

LibriTTS-R [1] is a sound quality improved version of the LibriTTS corpus (http://www.openslr.org/60/) which is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, published in 2019. The constituent samples of LibriTTS-R are identical to those of LibriTTS, with only the sound quality improved. To improve sound quality, a speech restoration model, Miipher proposed by Yuma Koizumi [2], was used.

For more information, refer to the paper [1]. If you use the LibriTTS-R corpus in your work, please cite the dataset paper [1] where it was introduced.

Audio samples of the ground-truth and TTS generated samples are available at the demo page: https://google.github.io/df-conformer/librittsr/

[1] Yuma Koizumi, Heiga Zen, Shigeki Karita, Yifan Ding, Kohei Yatabe, Nobuyuki Morioka, Michiel Bacchiani, Yu Zhang, Wei Han, and Ankur Bapna, "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus," arXiv, 2023.
[2] Yuma Koizumi, Heiga Zen, Shigeki Karita, Yifan Ding, Kohei Yatabe, Nobuyuki Morioka, Yu Zhang, Wei Han, Ankur Bapna, and Michiel Bacchiani, "Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations," arXiv, 2023.

"""
from os import makedirs
import sys
# sys.path.append("/mnt/users/jiazhijun/valle_23_4_22")
from pathlib import Path
from typing import Dict, Optional, Union
import os
current_working_directory = os.getcwd()
from tqdm import tqdm
print("Current working directory:", current_working_directory)  
sys.path.append(current_working_directory)
from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike

SPEAKER_DESCRIPTION = """
|Speaker|Gender|Native Language|# Wav Files|# Annotations|
|---|---|---|---|---|
|ABA|M|Arabic|1129|150|
|SKA|F|Arabic|974|150|
|YBAA|M|Arabic|1130|149|
|ZHAA|F|Arabic|1132|150|
|BWC|M|Chinese|1130|150|
|LXC|F|Chinese|1131|150|
|NCC|F|Chinese|1131|150|
|TXHC|M|Chinese|1132|150|
|ASI|M|Hindi|1131|150|
|RRBI|M|Hindi|1130|150|
|SVBI|F|Hindi|1132|150|
|KSP|M|Hindi|1132|150|
|TNI|F|Hindi|1131|150|
|HJK|F|Korean|1131|150|
|HKK|M|Korean|1131|150|
|YDCK|F|Korean|1131|150|
|YKWK|M|Korean|1131|150|
|EBVS|M|Spanish|1007|150|
|ERMS|M|Spanish|1132|150|
|MBMPS|F|Spanish|1132|150|
|NJS|F|Spanish|1131|150|
|HQTV|M|Vietnamese|1132|150|
|PNV|F|Vietnamese|1132|150|
|THV|F|Vietnamese|1132|150|
|cmu_us_jmk_arctic|M|Canadian|1132|150|
|cmu_us_awb_arctic|M|Scottish|1132|150|
|cmu_us_aew_arctic|M|may_native|1132|150|
|cmu_us_ahw_arctic|M|may_native|1132|150|
|cmu_us_bdl_arctic|M|may_native|1132|150|
|cmu_us_eey_arctic|F|may_native|1132|150|
|cmu_us_clb_arctic|F|may_native|1132|150|
|cmu_us_fem_arctic|M|may_native|1132|150|
|cmu_us_ljm_arctic|F|may_native|1132|150|
|cmu_us_lnh_arctic|F|may_native|1132|150|
|cmu_us_rms_arctic|M|may_native|1132|150|
|cmu_us_rxr_arctic|M|may_native|1132|150|
|cmu_us_slt_arctic|F|may_native|1132|150|
|cmu_us_ksp_arctic|M|Hindi|1132|150|
|TLV|M|Vietnamese|1132|150|
|suitcase_corpus|F|no_use|1132|150|
|**Total**|||**26867**|**3599**|"""

def _parse_speaker_description():
    meta = {}
    for line in SPEAKER_DESCRIPTION.splitlines()[3:-1]:
        _, spk, gender, native_lang, *_ = line.split("|")
        meta[spk.lower()] = {"gender": gender, "native_lang": native_lang}
    return meta

def prepare_l2_arctic(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares and returns the L2 Arctic manifests which consist of Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict with keys "read" and "spontaneous".
        Each hold another dict of {'recordings': ..., 'supervisions': ...}
    """
    corpus_dir_bef = Path(corpus_dir)
    # train_val_test = ["arctic_native_and_l2_arctic_accent"]

    corpus_dir = corpus_dir_bef
    print(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    speaker_meta = _parse_speaker_description()

    recordings = RecordingSet.from_recordings(
        # Example ID: zhaa-arctic_b0126
        Recording.from_file(
            wav, recording_id=f"{wav.parent.parent.parent.name.lower()}-{wav.stem}"
        )
        for wav in corpus_dir.rglob("*.wav")
    )
    supervisions = []
        # # with txt
        # for path in corpus_dir.rglob("*.txt"):
        #     # One utterance (line) per file
        #     text = path.read_text().strip()
        #     speaker = (
        #         path.parent.parent.name.lower()
        #     )  # <root>/ABA/transcript/arctic_a0051.txt -> aba
        #     # if is_suitcase_corpus:
        #     #     speaker = path.stem  # <root>/suitcase_corpus/transcript/aba.txt -> aba
        #     seg_id = (
        #     f"{speaker}-{path.stem}"
        #     )
        #     supervisions.append(
        #         SupervisionSegment(
        #             id=seg_id,
        #             recording_id=seg_id,
        #             start=0,
        #             duration=recordings[seg_id].duration,
        #             text=text,
        #             language="English",
        #             speaker=speaker,
        #             gender=speaker_meta[speaker]["gender"],
        #             custom={"accent": speaker_meta[speaker]["native_lang"]},
        #         )
        #     )
        # without txt
    for path in corpus_dir.rglob("*.wav"):
        
        text_path = path.parent.parent.joinpath("txt.done.data")

        name_to_find = str(path.stem) 
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:  
            for line in file:  
                if name_to_find in line:  
                    # 找到匹配的行，现在去除括号和多余的空格  
                    text = line.split('"')[1].strip()  
                    break  # 如果只需要找到第一个匹配项，找到后就可以退出循环  

        # One utterance (line) per file
        speaker = (
            path.parent.parent.parent.name.lower()
        )  # <root>/ABA/transcript/arctic_a0051.txt -> aba

        # if is_suitcase_corpus:
        #     speaker = path.stem  # <root>/suitcase_corpus/transcript/aba.txt -> aba
        seg_id = (
        f"{speaker}-{path.stem}"
        )
        supervisions.append(
            SupervisionSegment(
                id=seg_id,
                recording_id=seg_id,
                start=0,
                duration=recordings[seg_id].duration,
                text=text,
                language="English",
                speaker=speaker,
                custom={"accent": 'Hindi'},
            )
        )
    supervisions = SupervisionSet.from_segments(supervisions)

    validate_recordings_and_supervisions(recordings, supervisions)

    splits = {
        "read": {
            "recordings": recordings.filter(lambda r: "suitcase_corpus" not in r.id),
            "supervisions": supervisions.filter(
                lambda s: "suitcase_corpus" not in s.recording_id
            ),
        }
    }
    part="all"

    if output_dir is not None:
        output_dir = Path(output_dir)
        makedirs(output_dir, exist_ok=True)
        for key, manifests in splits.items():
            manifests["recordings"].to_file(
                output_dir / f"Indic_TTS_recordings_{part}.jsonl.gz"
            )
            manifests["supervisions"].to_file(
                output_dir / f"Indic_TTS_supervisions_{part}.jsonl.gz"
            )

    return splits



if __name__=="__main__":
    #need to rewrite
    prepare_l2_arctic("/scratch/indian_accent_datasets/indictts-english/IndicTTS", "/scratch/indian_accent_datasets/indictts-english/IndicTTS_lhotse")