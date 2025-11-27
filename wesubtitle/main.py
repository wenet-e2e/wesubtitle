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
    boxes, texts = ocr_results['rec_polys'], ocr_results['rec_texts']
    assert len(boxes) == len(texts)
    num_boxes = len(boxes)
    while idx < num_boxes:
        box, text = boxes[idx], texts[idx]
        # We assume the subtitle is at bottom of the video
        if box[0][1] < h * 0.75:
            idx += 1
            continue
        idx += 1
        con_box = copy.deepcopy(box)
        con_text = text
        while idx < num_boxes:
            n_box, n_text = boxes[idx], texts[idx]
            if abs(n_box[0][1] - box[0][1]) < h * 0.01 and \
            abs(n_box[3][1] - box[3][1]) < h * 0.01:
                con_box[1] = n_box[1]
                con_box[2] = n_box[2]
                con_text = con_text + ' ' + n_text
                idx += 1
            else:
                break
        candidates.append((con_box, con_text))
    # TODO(Binbin Zhang): Only support horion center subtitle
    if len(candidates) > 0:
        sub_boxes, subtitle = candidates[-1]
        # 计算字幕区域的中心 x 坐标
        subtitle_center_x = (sub_boxes[0][0] + sub_boxes[1][0]) / 2
        # 计算屏幕中心 x 坐标
        screen_center_x = w / 2
        # 检查字幕区域偏离中心位置不超过 20%
        offset_ratio = abs(subtitle_center_x - screen_center_x) / w
        if offset_ratio <= 0.20:
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
    parser.add_argument(
        '-m',
        '--max_duration',
        type=float,
        default=10.0,
        help=
        'maximum duration (in seconds) for the same subtitle, force re-detection if exceeded'
    )
    parser.add_argument('--model_type',
                        choices=['server', 'mobile'],
                        default='mobile',
                        help='OCR model type: server or mobile (default: mobile)')
    parser.add_argument('input_video', help='input video file')
    parser.add_argument('output_srt', help='output srt file')
    parser.add_argument('output_txt', nargs='?', default=None, help='output txt file (optional)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # 根据模型类型构建模型名称
    det_model_name = f"PP-OCRv5_{args.model_type}_det"
    rec_model_name = f"PP-OCRv5_{args.model_type}_rec"
    ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_detection_model_name=det_model_name,
                    text_recognition_model_name=rec_model_name)
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
            # 检查字幕持续时间是否超过最大限制
            current_duration = (cur - start) / fps
            if current_duration > args.max_duration:
                # 强制重新检测
                _add_subs(cur - args.subsampling)
                detected = False
                continue
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

    # Write txt file (if output_txt is provided)
    if args.output_txt:
        with open(args.output_txt, 'w', encoding='utf8') as fout:
            for idx in range(len(subs)):
                # same subtitle occurs once
                if idx > 0:
                    if subs[idx].content == subs[idx - 1].content:
                        continue
                fout.write(subs[idx].content + "\n")


if __name__ == '__main__':
    main()
