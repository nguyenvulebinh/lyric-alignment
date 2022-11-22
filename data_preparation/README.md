# Vietnamese song data preparation

## Data crawler

This folder contains a list of song name and code used for downloading data (mp3 tracks with lyric file) from the zingmp3 website. 

```bash
wget https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp

pip install requests tqdm

python zingmp3_crawler.py
```

After crawling the data, you need to convert mp3 to wav file (16k, single channel) using ffmpeg. This converting job can be done by [yt-dlp](https://github.com/yt-dlp/yt-dlp) but I have yet to try.

```bash
python convert_wav.py
```

The result in the download folder should look like this

```
...
yeu-vo-lynk-lee.lrc
yeu-vo-lynk-lee.mp3
yeu-vo-lynk-lee.wav
yeu-vo-nhat-remix-ly-tuan-kiet-dj-minh-ly.lrc
yeu-vo-nhat-remix-ly-tuan-kiet-dj-minh-ly.mp3
yeu-vo-nhat-remix-ly-tuan-kiet-dj-minh-ly.wav
yeu-vo-nhat-remix-ly-tuan-kiet.lrc
yeu-vo-nhat-remix-ly-tuan-kiet.mp3
yeu-vo-nhat-remix-ly-tuan-kiet.wav
yeu-vo-thuong-con-nguyen-hoang-phong.lrc
yeu-vo-thuong-con-nguyen-hoang-phong.mp3
yeu-vo-thuong-con-nguyen-hoang-phong.wav
...
```

A sample of lrc file:

```
[ar: Lee7 ft. Lynk Lee]
[ti: Ngày Rồi Yêu]
[id: puskkhsi]
[00:00.00]Bài Hát: 10 Ngày Yêu Rồi
[00:00.00]Ca Sĩ: Lee7 ft. Lynk Lee
[00:00.67]Ánh mắt 
[00:01.23]Dịu dàng nhìn 
[00:01.86]Anh đầy yêu thương 
[00:02.61]Anh trao tình yêu
[00:03.52]
[00:03.92]Vòng tay khẽ 
[00:04.73]Nới rộng để người 
[00:05.42]ôm anh
[00:05.98]Hôn anh thật nhiều
[00:07.12]Chợt gió 
[00:07.61]Vuốt tóc mềm của người
[00:08.48]Và gió 
[00:09.19]để mùi hương 
[00:09.73]Trên mái tóc em 
[00:10.90]Nhẹ bay 
[00:11.40]Trên mây kìa baby
[00:13.08]
[00:13.90]Ah, baby
[00:15.08]Yêu em từ cái 
...
```

## Process lyric files

Split lyric to smaller segment.

```
python split_lyric.py
```

- Random split 10-20s
- Remove noise segment text
- ...

Here is the result of a lyric file after split

```json
{
    "sid": "yeu-vo-lynk-lee",
    "lyric": [
        [
            [
                2610,
                3920,
                "Anh trao tình yêu"
            ],
            [
                3920,
                4730,
                "Vòng tay khẽ"
            ],
            [
                4730,
                5420,
                "Nới rộng để người"
            ],
            [
                5420,
                5980,
                "ôm anh"
            ],
            [
                5980,
                7120,
                "Hôn anh thật nhiều"
            ],
            [
                7120,
                7610,
                "Chợt gió"
            ],
            [
                7610,
                8480,
                "Vuốt tóc mềm của người"
            ],
            [
                8480,
                9190,
                "Và gió"
            ],
            [
                9190,
                9730,
                "để mùi hương"
            ],
            [
                9730,
                10900,
                "Trên mái tóc em"
            ],
            [
                10900,
                11400,
                "Nhẹ bay"
            ],
            [
                11400,
                13900,
                "Trên mây kìa baby"
            ]
        ],
        [
            [
                13900,
                15080,
                "Ah, baby"
            ],
            [
                15080,
                15770,
                "Yêu em từ cái"
            ],
        ]
        ...
    ]
}
```

Because of lyric came from diverse of source, it can  include special characters, mix English and Vietnamese word, number format (date, time, currency,...), nickname, ... This kind of data will make the model hard to map between audio signal and text lyric. Our soulution is mapping all word of lyric from written form to spoken form. For example:

| Written                                           | Spoken      |
|--------------------------------------------------|--------------|
| joker | giốc cơ |
| running                  | răn ninh    |
| 0h   | không giờ   |

To convert English words to pronunciation way in Vietnamese, we use [nguyenvulebinh/spelling-oov](
https://huggingface.co/nguyenvulebinh/spelling-oov) model. For handling number format, we use [Vinorm](https://github.com/v-nhandt21/Vinorm). For other special characters ".,?...", we delete it.

*Good news: we have already mapped most of the words in the original lyric for you. The result is saved in the **map_dict.txt** file*

The next step (clean_lyric.py) is simply to use **map_dict.txt** to normalize the lyric file. It will output a json file, the collection of all tracks and all text segments formatted. 

```bash
python clean_lyric.py

```

## Format all the things to huggingface dataset format

Although all tracks have been split into smaller segments, we still need to split the audio into smaller audio. The script split_audio_by_segment.py will do that.

```bash
python split_audio_by_segment.py
```

The script will make a new folder and put all split audio and lyrics into that folder. The folder structure is like the one below.

```
formated_data
    |---yeu-vo-lynk-lee
    |    |
    |    yeu-vo-lynk-lee_0.txt
    |    yeu-vo-lynk-lee_0.wav
    |    yeu-vo-lynk-lee_10.txt
    |    yeu-vo-lynk-lee_10.wav
    |    yeu-vo-lynk-lee_11.txt
    |    yeu-vo-lynk-lee_11.wav
    |    yeu-vo-lynk-lee_12.txt
    |    yeu-vo-lynk-lee_12.wav
    |    yeu-vo-lynk-lee_13.txt
    |    yeu-vo-lynk-lee_13.wav
    |    ...
    |---yeu-vo-nhat-remix-ly-tuan-kiet-dj-minh-ly
    |    |
    |    ...
    ...

```

A sample of lyric file
```
# yeu-vo-lynk-lee_0.txt
0       1310    anh trao tình yêu
1310    2120    vòng tay khẽ
2120    2810    nới rộng để người
2810    3370    ôm anh
3370    4510    hôn anh thật nhiều
4510    5000    chợt gió
5000    5870    vuốt tóc mềm của người
5870    6580    và gió
6580    7120    để mùi hương
7120    8290    trên mái tóc em
8290    8790    nhẹ bay
8790    11290   trên mây kìa bây bi
```

In the final step, we run the script make_data.py to collect all audio and lyrics and put them into a single file following the [huggingface dataset](https://huggingface.co/docs/datasets/index) format.

```bash
python make_data.py
```

The result after that can be load using [load_from_disk](https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_from_disk) function. For our case, it will look like below:

```
Dataset({
    features: ['audio', 'segment_text', 'segment_align', 'sid'],
    num_rows: 355337
})
```

*Good news: We have already uploaded all data to huggingface dataset repository [nguyenvulebinh/song_dataset](https://huggingface.co/datasets/nguyenvulebinh/song_dataset)* 

**All data is subject to website policy https://zingmp3.vn**