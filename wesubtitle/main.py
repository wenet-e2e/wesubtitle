import argparse
import copy
import datetime

import cv2
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity
import srt


def box2int(box):
    for i in range(len(box)):
        for j in range(len(box[i])):
            box[i][j] = int(box[i][j])
    return box


def detect_subtitle_area(ocr_results, h, w):
    '''
    Args:
        w(int): width of the input video
        h(int): height of the input video
    '''
    ocr_results = ocr_results[0]  # 0, the first image result
    # Merge horizon text areas
    idx = 0
    candidates = []
    while idx < len(ocr_results):
        boxes, text = ocr_results[idx]
        # We assume the subtitle is at bottom of the video
        if boxes[0][1] < h * 0.75:
            idx += 1
            continue
        idx += 1
        con_boxes = copy.deepcopy(boxes)
        con_text = text[0]
        while idx < len(ocr_results):
            n_boxes, n_text = ocr_results[idx]
            if abs(n_boxes[0][1] - boxes[0][1]) < h * 0.01 and \
               abs(n_boxes[3][1] - boxes[3][1]) < h * 0.01:
                con_boxes[1] = n_boxes[1]
                con_boxes[2] = n_boxes[2]
                con_text = con_text + ' ' + n_text[0]
                idx += 1
            else:
                break
        candidates.append((con_boxes, con_text))
    # TODO(Binbin Zhang): Only support horion center subtitle
    if len(candidates) > 0:
        sub_boxes, subtitle = candidates[-1]
        # offset is less than 10%
        if (sub_boxes[0][0] + sub_boxes[1][0]) / w > 0.90:
            return True, box2int(sub_boxes), subtitle
    return False, None, None


def get_args():
    parser = argparse.ArgumentParser(description='we subtitle')
    parser.add_argument('-s',
                        '--subsampling',
                        type=int,
                        default=3,
                        help='subsampling rate, for speedup')
    parser.add_argument('-t',
                        '--similarity_thresh',
                        type=float,
                        default=0.8,
                        help='similarity threshold')
    parser.add_argument('input_video', help='input video file')
    parser.add_argument('output_srt', help='output srt file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    cap = cv2.VideoCapture(args.input_video)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Video info w: {}, h: {}, count: {}, fps: {}'.format(
        w, h, count, fps))

    cur = 0
    detected = False
    box = None
    content = ''
    start = 0
    ref_gray_image = None
    subs = []

    def _add_subs(end):
        print('New subtitle {} {} {}'.format(start / fps, end / fps, content))
        subs.append(
            srt.Subtitle(
                index=0,
                start=datetime.timedelta(seconds=start / fps),
                end=datetime.timedelta(seconds=end / fps),
                content=content.strip(),
            ))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if detected:
                _add_subs(cur)
            break
        cur += 1
        if cur % args.subsampling != 0:
            continue
        if detected:
            # Compute similarity to reference subtitle area, if the result is
            # bigger than thresh, it's the same subtitle, otherwise, there is
            # changes in subtitle area
            hyp_gray_image = frame[box[1][1]:box[2][1], box[0][0]:box[1][0], :]
            hyp_gray_image = cv2.cvtColor(hyp_gray_image, cv2.COLOR_BGR2GRAY)
            similarity = structural_similarity(hyp_gray_image, ref_gray_image)
            if similarity > args.similarity_thresh:  # the same subtitle
                continue
            else:
                # Record current subtitle
                _add_subs(cur - args.subsampling)
                detected = False
        else:
            # Detect subtitle area
            ocr_results = ocr.ocr(frame)
            detected, box, content = detect_subtitle_area(ocr_results, h, w)
            if detected:
                start = cur
                ref_gray_image = frame[box[1][1]:box[2][1],
                                       box[0][0]:box[1][0], :]
                ref_gray_image = cv2.cvtColor(ref_gray_image,
                                              cv2.COLOR_BGR2GRAY)
    cap.release()

    # Write srt file
    with open(args.output_srt, 'w', encoding='utf8') as fout:
        fout.write(srt.compose(subs))


if __name__ == '__main__':
    main()
