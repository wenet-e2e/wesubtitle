# Copyright (c) 2026 Binbin Zhang (binbzha@qq.com)
#
# Adjust the boundary according to VAD/speaker information

import argparse
import datetime
import os
import wave

import numpy as np
import srt
import torch
import wespeaker


def read_srt(srt_path):
    """Read SRT file and return subtitle list"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    subtitles = list(srt.parse(srt_content))
    return subtitles


def read_wav(wav_path):
    """Read WAV file and return sample rate and audio data"""
    with wave.open(wav_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio_array = np.frombuffer(audio_data, dtype=dtype)

    # Keep only the first channel if multi-channel
    if n_channels > 1:
        audio_array = audio_array[::n_channels]

    return sample_rate, audio_array


def extract_audio_by_seconds(audio_array, sample_rate, start_sec, end_sec):
    """Extract audio segment by seconds

    Args:
        audio_array: numpy audio array
        sample_rate: sample rate
        start_sec: start time in seconds
        end_sec: end time in seconds

    Returns:
        Audio segment as numpy array
    """
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    start_sample = max(0, start_sample)
    end_sample = min(len(audio_array), end_sample)

    return audio_array[start_sample:end_sample]


def load_srt_and_wav(srt_path, wav_path):
    """Load SRT and WAV separately

    Args:
        srt_path: SRT file path
        wav_path: WAV file path

    Returns:
        subtitles: list of srt.Subtitle
        audio_array: numpy audio array
        sample_rate: sample rate
    """
    subtitles = read_srt(srt_path)
    sample_rate, audio_array = read_wav(wav_path)

    print(f"Loaded SRT with {len(subtitles)} subtitles")
    print(
        f"Loaded WAV: sample_rate={sample_rate}, duration={len(audio_array)/sample_rate:.2f}s"
    )

    return subtitles, audio_array, sample_rate


def subtitles_to_segments(subtitles, audio_array, sample_rate):
    """Convert subtitles to segments with audio

    Args:
        subtitles: list of srt.Subtitle
        audio_array: numpy audio array
        sample_rate: sample rate

    Returns:
        list of segment dicts
    """
    segments = []
    for sub in subtitles:
        start_sec = sub.start.total_seconds()
        end_sec = sub.end.total_seconds()
        audio_segment = extract_audio_by_seconds(audio_array, sample_rate,
                                                 start_sec, end_sec)
        segments.append({
            'index': sub.index,
            'start': start_sec,
            'end': end_sec,
            'text': sub.content,
            'audio': audio_segment,
        })
    return segments


def compute_speaker_embedding(speaker_model, audio, sample_rate):
    """Compute speaker embedding for audio segment

    Args:
        speaker_model: wespeaker model
        audio: numpy audio array
        sample_rate: sample rate

    Returns:
        speaker embedding as numpy array
    """
    # Normalize audio to float32 [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0

    embedding = speaker_model.extract_embedding_from_pcm(
        torch.from_numpy(audio).unsqueeze(0), sample_rate)
    return embedding


def merge_same_text_segments(segments, audio_array, sample_rate, max_gap):
    """Merge adjacent segments with identical text, including gap audio

    Args:
        segments: list of segment dicts
        audio_array: full audio array
        sample_rate: sample rate
        max_gap: max time gap (in seconds) between segments to merge

    Returns:
        list of merged segments
    """
    if not segments:
        return []

    merged = []
    current = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]
        # Time gap between current segment end and next segment start
        gap = seg['start'] - current['end']

        # Merge if same text and gap <= max_gap
        if seg['text'] == current['text'] and gap <= max_gap:
            # Extract audio from current start to seg end (including gap)
            current['end'] = seg['end']
            current['audio'] = extract_audio_by_seconds(
                audio_array, sample_rate, current['start'], current['end'])
        else:
            merged.append(current)
            current = seg.copy()

    merged.append(current)

    # Re-index
    for i, seg in enumerate(merged):
        seg['index'] = i + 1

    return merged


def merge_same_speaker_segments(segments, audio_array, sample_rate,
                                speaker_model, speaker_thresh, max_duration,
                                max_gap_duration):
    """Merge adjacent segments with same speaker, up to max_duration

    Args:
        segments: list of segment dicts
        audio_array: full audio array
        sample_rate: sample rate
        speaker_model: wespeaker model
        speaker_thresh: similarity threshold for same speaker
        max_duration: max duration after merging

    Returns:
        list of merged segments
    """
    if not segments:
        return []

    # Compute speaker embeddings for all segments
    print("Computing speaker embeddings...")
    for seg in segments:
        seg['embedding'] = compute_speaker_embedding(speaker_model,
                                                     seg['audio'], sample_rate)

    merged = []
    current = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]
        current_duration = current['end'] - current['start']
        new_duration = seg['end'] - current['start']
        gap = seg['start'] - current['end']

        # Compute similarity
        similarity = speaker_model.cosine_similarity(current['embedding'],
                                                     seg['embedding'])

        # Merge if same speaker and new duration <= max_duration
        if similarity >= speaker_thresh and new_duration <= max_duration and gap <= max_gap_duration:
            # Extend current segment (including gap audio)
            current['end'] = seg['end']
            current['text'] = current['text'] + ' ' + seg['text']
            current['audio'] = extract_audio_by_seconds(
                audio_array, sample_rate, current['start'], current['end'])
            # Update embedding to the merged audio
            current['embedding'] = compute_speaker_embedding(
                speaker_model, current['audio'], sample_rate)
        else:
            merged.append(current)
            current = seg.copy()

    merged.append(current)

    # Re-index and clean up
    for i, seg in enumerate(merged):
        seg['index'] = i + 1
        if 'embedding' in seg:
            del seg['embedding']

    return merged


def expand_segment_boundaries(segments, expand_sec, total_duration):
    """Expand segment boundaries without overlapping

    Args:
        segments: list of segment dicts with 'start', 'end'
        expand_sec: seconds to expand each boundary
        total_duration: total duration of audio in seconds

    Returns:
        list of segments with expanded boundaries
    """
    for i, segment in enumerate(segments):
        # Expand start, but not before 0 or previous segment's end
        prev_end = segments[i - 1]['end'] if i > 0 else 0
        new_start = max(prev_end, segment['start'] - expand_sec)
        new_start = max(0, new_start)

        # Expand end, but not after total duration or next segment's start
        next_start = segments[
            i + 1]['start'] if i < len(segments) - 1 else total_duration
        new_end = min(next_start, segment['end'] + expand_sec)
        new_end = min(total_duration, new_end)

        segment['start'] = new_start
        segment['end'] = new_end

    return segments


def write_textgrid(segments, output_path, total_duration):
    """Write segments to TextGrid file

    Args:
        segments: list of segment dicts with 'start', 'end', 'text'
        output_path: output TextGrid file path
        total_duration: total duration of audio in seconds
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # TextGrid header
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n')
        f.write('\n')
        f.write(f'xmin = 0\n')
        f.write(f'xmax = {total_duration}\n')
        f.write('tiers? <exists>\n')
        f.write('size = 1\n')
        f.write('item []:\n')

        # Interval tier
        f.write('    item [1]:\n')
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "text"\n')
        f.write(f'        xmin = 0\n')
        f.write(f'        xmax = {total_duration}\n')
        f.write(f'        intervals: size = {len(segments)}\n')

        for i, seg in enumerate(segments):
            f.write(f'        intervals [{i + 1}]:\n')
            f.write(f'            xmin = {seg["start"]}\n')
            f.write(f'            xmax = {seg["end"]}\n')
            # Escape quotes in text
            text = seg['text'].replace('"', '""')
            f.write(f'            text = "{text}"\n')

    print(f"Written TextGrid to {output_path}")


def write_srt(segments, output_path):
    """Write segments to SRT file

    Args:
        segments: list of segment dicts with 'start', 'end', 'text'
        output_path: output SRT file path
    """
    subtitles = []
    for i, seg in enumerate(segments):
        subtitles.append(
            srt.Subtitle(
                index=i + 1,
                start=datetime.timedelta(seconds=seg['start']),
                end=datetime.timedelta(seconds=seg['end']),
                content=seg['text'].strip(),
            ))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))

    print(f"Written SRT to {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description='Load SRT and WAV segments')
    parser.add_argument('srt_file', help='input SRT file')
    parser.add_argument('wav_file', help='input WAV file')
    parser.add_argument('--max_duration',
                        type=float,
                        default=20,
                        help='max audio duration')
    parser.add_argument('--max_gap_duration',
                        type=float,
                        default=1,
                        help='max gap duration between segments to merge')
    parser.add_argument('--speaker_thresh',
                        type=float,
                        default=0.60,
                        help='similarity threshold for same speaker')

    parser.add_argument('--max_same_text_duration',
                        type=float,
                        default=0.5,
                        help='max same text duration')
    parser.add_argument('--expand_boundary',
                        type=float,
                        default=0.6,
                        help='expand boundary (in seconds) for each segment')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    print(f"Loading {args.srt_file} and {args.wav_file}")

    # Load SRT and WAV
    subtitles, audio_array, sample_rate = load_srt_and_wav(
        args.srt_file, args.wav_file)

    print("Read all segments from SRT")
    segments = subtitles_to_segments(subtitles, audio_array, sample_rate)
    print(f"Original segments: {len(segments)}")

    print("Merge adjacent segments with identical text (including gap audio)")
    segments = merge_same_text_segments(segments, audio_array, sample_rate,
                                        args.max_same_text_duration)
    print(f"After merging same text: {len(segments)}")

    # Load wespeaker model
    print("Loading wespeaker model...")
    speaker_model = wespeaker.load_model('campplus')

    print(
        "Merge adjacent segments with same speaker embedding, up to max_duration"
    )
    segments = merge_same_speaker_segments(segments, audio_array, sample_rate,
                                           speaker_model, args.speaker_thresh,
                                           args.max_duration,
                                           args.max_gap_duration)
    print(f"After merging same speaker: {len(segments)}")

    print(f"Expand boundary by {args.expand_boundary}s for each segment")
    total_duration = len(audio_array) / sample_rate
    segments = expand_segment_boundaries(segments, args.expand_boundary,
                                         total_duration)

    print("Write TextGrid file with same prefix as wav file")
    wav_dir = os.path.dirname(args.wav_file)
    wav_basename = os.path.splitext(os.path.basename(args.wav_file))[0]
    textgrid_path = os.path.join(wav_dir, wav_basename + '.TextGrid')
    write_textgrid(segments, textgrid_path, total_duration)

    print("Write new srt file with same prefix as wav file")
    srt_path = os.path.join(wav_dir, wav_basename + '.adjust.srt')
    write_srt(segments, srt_path)


if __name__ == '__main__':
    main()
