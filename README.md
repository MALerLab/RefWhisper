# RefWhisper

This is a repository for our paper on DLfM 2023, "Aligning Incomplete Lyrics of Korean Folk Song Dataset using Whisper".

The original dataset is available at [here](http://urisori.co.kr/urisori-en/doku.php?id=start)

## Requirements
We recommend ``pipenv`` for installing the requirements. The requirements are given in ``Pipfile``. Python 3.11 is required (Pillow 13 needs Python >= 3.11, and torch 2.0.1 supports up to 3.11).
### Install pipenv and requirements
```
pip install pipenv
pipenv install
pipenv shell
```

## Model
The trained model is available on [Hugging Face](https://huggingface.co/MALer-Lab/refwhisper) (`ref_whisper_weight.pt`, 11.3 GB, the same file that was previously shared on OneDrive).
```
wget https://huggingface.co/MALer-Lab/refwhisper/resolve/main/ref_whisper_weight.pt
```

## Usage
Run the commands from the root of this repository. The reference lyric is a plain text file with one lyric line per line; the line breaks are used for the line-level alignment.

### Command line
```
python -m ref_whisper.inference_timestamped song.wav song.txt --model ref_whisper_weight.pt --language ko --output_dir results
```
You can also pass a directory of `.wav` files and a directory of `.txt` files; they are paired by file name. For each audio file, `results/` gets `{name}.words.json` together with `.txt`, `.srt`, `.vtt`, `.csv` and `.tsv` outputs. The JSON holds the word timestamps with the matched reference words (`segments[].words`) and the line-level alignment of each reference line (`ref_align_segments`). Without `--output_dir`, the JSON is printed to stdout.

### Python
```python
from ref_whisper.inference_timestamped import load_model, transcribe_timestamped

model = load_model("ref_whisper_weight.pt", device="cuda")

# transcription guided by the reference lyric
result = model.transcribe(audio="song.wav", reference_text="song.txt", temperature=0, language="ko")
print(result["text"])

# transcription with word timestamps and line-level alignment to the reference lyric
result = transcribe_timestamped(model, "song.wav", "song.txt", language="ko")
print(result["ref_align_segments"])
```
`reference_text` takes either a path to the lyric file or the lyric text itself.


## Anthology of Korean Traditional Folksongs
[Official website](http://urisori.co.kr/urisori-en/doku.php?id=start)

The metadata we collected from the website is available in [Finding Tori](https://github.com/danbinaerinHan/finding-tori) `metadata.csv`, including the url for each song. You can download the audio files from the website.


## Lyric Transcribed Result
The transcribed result is provided in ``transcription_result_csvs.tar.gz``. You can untar the file by running ``tar -xvzf transcription_result_csvs.tar.gz``.

The transcription is given in three files:
- ``{song_id}_transcribed.txt``
  - The transcribed lyric of the song
- ``{song_id}_word_align.csv``
  - Word-level alignment between the transcribed lyric and the reference lyric
- ``{song_id}_ref_align.csv``
  - Line-level alignment between the reference lyric and the transcribed lyric
  - In this file, there are two alignment. 
    - 1) Serial alignment between the transcribed and the reference. In this alignment, the two lyrics are aligned in serial order from the beginning to the end.
    - 2) Open-beginning/open-end alignment between each line of the reference lyric and the entire corresponding transcription. This trys to find the optimal alignment for that specific lyric line, not considering the alignment for entire lyrics.

The word-level alignment and the open-beginning/open-end line alignment correspond to `segments[].words` and `ref_align_segments` in the command-line output. The serial alignment and the `cer` column were computed by a post-processing step that is not included in this repository.

## Citation
If you use this code or the dataset, please cite our paper.
```bibtex
@inproceedings{han2023aligning,
  title     = {Aligning Incomplete Lyrics of Korean Folk Song Dataset using Whisper},
  author    = {Han, Danbinaerin and Kim, Daewoong and Jeong, Dasaem},
  booktitle = {Proceedings of the 10th International Conference on Digital Libraries for Musicology (DLfM)},
  pages     = {7--11},
  year      = {2023},
  address   = {Milan, Italy},
  publisher = {ACM},
  doi       = {10.1145/3625135.3625154}
}
```
