# 调整字幕分段和边界

考虑获取字幕后，我们最终的字幕要给 ASR 和 TTS 任务使用，所以需要对字幕进一步处理以适合这些任务。
需要考虑的处理有:
1. 合并临近相同说话人的句子，避免因字幕太碎导致的句子被切的多为短句的问题。
2. 对句子边界做扩展，尽可能避免字幕标注时间不准导致的标准不准的问题。


对于单个视频，具体处理过程如下：

1. 使用 ffmpeg 提取视频对应的音频

```
ffmpeg -i test.mp4  -vn -acodec pcm_s16le -ar 16000 -ac 1 test.wav
```

2. 使用 deepfilternet 对音频进行降噪处理，其安装见 [install](https://github.com/Rikorose/DeepFilterNet?tab=readme-ov-file#usage)

```
deep-filter test.wav # 降噪后的音频见 out/test.wav
```

3. 使用 wesubtitle/adjust.py 对音频分段进行合并，并扩展边界

```
pip install wespeaker srt
```

```
python wesubtitle/adjust.py test.srt out/test.wav
```

其中:
* out/test.wav 为降噪后的音频
* test.srt 为 OCR 生成的字幕文件

该脚本会在输入 wav 文件同目录下生成两个同名文件:
* test.adjust.srt: 调整后的字幕文件
* test.adjust.TextGrid: 调整后的字幕时间戳文件