# RefWhisper

This is a repository for our paper on DLfM 2023, "Aligning Incomplete Lyrics of Korean Folk Song Dataset using Whisper".

The original dataset is available at [here](http://urisori.co.kr/urisori-en/doku.php?id=start)

## Requirements
We recommend ``pipenv`` for installing the requirements. The requirements are given in ``Pipfile``.
### Install pipenv and requirements
```
pip install pipenv
pipenv install
pipenv shell
```

## Model
The pre-trained model is provided in [OneDrive](https://sogang365-my.sharepoint.com/:u:/g/personal/dasaem_jeong_o365_sogang_ac_kr/ESVuo8AO37tCq9_xT7P9QzABoggFDq-qLMKQ61ZOMnjdpw?e=gugvgQ)


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

## Citation
If you use this code or the dataset, please cite our paper.
```
