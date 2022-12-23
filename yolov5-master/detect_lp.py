import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import cv2
import pandas
model = torch.hub.load('ultralytics/yolov5', 'custom','E:/Study/Python_Yolo_Dataset/NumberPlate_Recognition_DataSet/yolov5-master/runs/train/exp/weights/best.pt')  # or yolov5m, yolov5l, yolov5x, etc.
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/best.pt')  # custom trained model
model2 = torch.hub.load('ultralytics/yolov5', 'custom','E:/Study/Python_Yolo_Dataset/NumberPlate_Recognition_DataSet/yolov5-master/runs/train/exp2/weights/best.pt')

font_scale=0.5
thickness=1

alphabet=[  "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "M",
            "T",
            "V",
            "X",
            "P",
            "U",
            "L",
            "N",
            "S",
            'K',
            "R"]
# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

thresh_iou = 0.03
def detect_digit(frame):
    result=model(frame)
    if not result.xyxy[0].size(dim=0) == 0:
        boxes=result.xyxy[0]  # im1 predictions (tensor)
        result.pandas().xyxy[0] # im1 predictions (pandas)
        # print(result.pandas().xyxy[0] )
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()

        keep = []

        while len(order) > 0:
            idx = order[-1]

            keep.append(boxes[idx])
            order = order[:-1]

            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)

            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            w = xx2 - xx1
            h = yy2 - yy1

            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            inter = w * h

            rem_areas = torch.index_select(areas, dim=0, index=order)
            union = (rem_areas - inter) + areas[idx]
            IoU = inter / union

            mask = IoU < thresh_iou
            order = order[mask]
        #conver tensor to Dataframe
        px = pandas.DataFrame(keep).astype("float")
        info =  px.to_dict(orient="records")  # Predicciones
        list_info=[]
        if len(keep)!=0:
            for result in info:
                # print(result)
                conf=  result[4]
                if conf >= 0.4:
                    # Classes
                    cls = int(result[5])
                    # Xi
                    xA = int(result[0])
                    # Yi
                    yA = int(result[1])
                    # Xf
                    xB = int(result[2])
                    # Yf
                    yB = int(result[3])

                    list_info.append([cls,int(conf),int(xA),int(yA),int(xB-xA),int(yB-yA)])
                    if cls>=10 and cls <=28:
                        cv2.putText(frame, alphabet[cls-10], (xA, yA-3), cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_scale, color=(0, 0, 255), thickness=thickness)
                    else:
                        cv2.putText(frame, str(cls), (xA, yA - 3), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                                    color=(0, 0, 255), thickness=thickness)
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # for obj in list_info:
            # print(obj.xA, obj.yA,obj.yA,obj.xB,obj.yB,obj.cls,obj.conf, sep=' ')
            (license_plate_text, correct_license_plate)=arrange_digit(list_info)
            print(license_plate_text)
        cv2.imshow('Digit', frame)



threshole=0.6
def arrange_digit(list_info):
    number_of_character = len(list_info)
    if number_of_character == 0:
        return ('0', 'False')
    sum_y = 0
    for i in range(number_of_character):
        sum_y += list_info[i][3]
    average_y = sum_y / number_of_character
    arr_upper, arr_lower = [], []
    for i in range(number_of_character):
        if list_info[i][3] < average_y:
            arr_upper.append(list_info[i])
        else:
            arr_lower.append(list_info[i])

    number_of_upper_character = len(arr_upper)
    number_of_lower_character = len(arr_lower)

    for i in range(number_of_upper_character):
        already_sorted = True
        for j in range(number_of_upper_character - i - 1):
            if arr_upper[j][2] > arr_upper[j + 1][2]:
                arr_upper[j], arr_upper[j + 1] = arr_upper[j + 1], arr_upper[j]
                already_sorted = False
        if already_sorted:
            break

    for i in range(number_of_lower_character):
        already_sorted = True
        for j in range(number_of_lower_character - i - 1):
            if arr_lower[j][2] > arr_lower[j + 1][2]:
                arr_lower[j], arr_lower[j + 1] = arr_lower[j + 1], arr_lower[j]
                already_sorted = False
        if already_sorted:
            break

    license_plate_text = ''
    for i in range(number_of_upper_character):
        if arr_upper[i][0] >=10 and arr_upper[i][0] <=28:
            arr_upper[i][0]=alphabet[arr_upper[i][0]-10]
            license_plate_text += str(arr_upper[i][0])
        else:
            license_plate_text += str(arr_upper[i][0])
    license_plate_text += '-'
    for i in range(number_of_lower_character):
        if arr_lower[i][0] >=10 and arr_lower[i][0] <=28:
            arr_lower[i][0]=alphabet[arr_upper[i][0]-10]
            license_plate_text += str(arr_lower[i][0])
        else:
            license_plate_text += str(arr_lower[i][0])

    correct_license_plate = 'False'
    if number_of_upper_character == 4 and number_of_lower_character == 4 and abs(
            arr_upper[0][2] - arr_lower[0][2]) < threshole:
        correct_license_plate = 'True'
    if number_of_upper_character == 4 and number_of_lower_character == 5 and abs(
            arr_upper[0][2] - arr_lower[0][2]) >= threshole:
        correct_license_plate = 'True'

    return (license_plate_text, correct_license_plate)


def detect_lp(source_frame):
    result=model2(source_frame)
    result.xyxy[0]  # im1 predictions (tensor)
    result.pandas().xyxy[0]  # im1 predictions (pandas)
    crop_img=source_frame

    if not result.xyxy[0].size(dim=0) == 0:
        for box in result.xyxy[0]:
            if box[5] == 0 and box[4]>0.8:
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])
                crop_img = source_frame[yA:yB, xA:xB]
                cv2.rectangle(source_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    return crop_img,source_frame




def detect_camera():
    vid = cv2.VideoCapture(0)

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        crop_img, source_frame = detect_lp(frame)
        cv2.imshow('crop_img', crop_img)
        cv2.imshow('source_frame', source_frame)
        detect_digit(crop_img)
        # cv2.imshow('frame', frame)
        cv2.waitKey(1)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
def detect_image():
    path = r'E:/Study/Python_Yolo_Dataset/NumberPlate_Recognition_DataSet/yolov5-master/data/images/lp (6).jpg'
    frame=cv2.imread(path)
    crop_img, source_frame = detect_lp(frame)
    cv2.imshow('crop_img', crop_img)
    cv2.imshow('source_frame', source_frame)
    digit = detect_digit(crop_img)
    cv2.imshow('digit', digit)
    # cv2.imshow('frame', frame)
    cv2.waitKey()

def detect_video():
    cap = cv2.VideoCapture(r'E:/Study/Python_Yolo_Dataset/NumberPlate_Recognition_DataSet/yolov5-master/data/images/video.mp4')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            crop_img, source_frame = detect_lp(frame)
            detect_digit(crop_img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
if __name__ == '__main__':
    detect_camera()