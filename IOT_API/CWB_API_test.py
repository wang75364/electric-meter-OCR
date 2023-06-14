import lineTool
import requests
import cv2
import os
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\UserProgram\tesseract_ocr\tesseract.exe"

#讀取模型與訓練權重
def initNet():
    #'yolov4-tiny-myobj.cfg'
    CONFIG = os.path.join('yolov4-tiny-myobj.cfg')
    WEIGHT = os.path.join('backup','yolov4-tiny-myobj_best.weights')
    # WEIGHT = './train_finished/yolov4-tiny-myobj_last.weights'
    net = cv2.dnn.readNet(CONFIG, WEIGHT)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255.0)
    model.setInputSwapRB(True)
    return model

#物件偵測
def nnProcess(image, model):
    classes, confs, boxes = model.detect(image, 0.4, 0.1)
    return classes, confs, boxes

#框選偵測到的物件，並裁減
def drawBox(image, classes, confs, boxes):
    new_image = image.copy()
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 18 < 0:
            x = 18
        if y - 18 < 0:
            y = 18
        cv2.rectangle(new_image, (x - 18, y - 18), (x + w + 20, y + h + 24), (0, 255, 0), 3)
    return new_image

# 裁減圖片
def cut_img(image, classes, confs, boxes):
    cut_img_list = []
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box
        if x - 18 < 0:
            x = 18
        if y - 18 < 0:
            y = 18
        cut_img = image[y - 18:y + h + 20, x - 18:x + w + 25]
        cut_img_list.append(cut_img)
    return cut_img_list[0]

# 儲存已完成前處理之圖檔(中文路徑)
def saveClassify(image, output):
    cv2.imencode(ext='.jpg', img=image)[1].tofile(output)

def read_class(image,class_list,confs_list,box_list,object):
    new_image=image.copy
    for (classid, conf, box) in zip(class_list, confs_list, box_list):
        if classid == object:
            new_image = image.copy()
            x, y, w, h = box
            if x - 18 < 0:
                x = 18
            if y - 18 < 0:
                y = 18
            cv2.putText(new_image,str(classid),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(new_image, (x - 18, y - 18), (x + w + 20, y + h + 24), (0, 255, 0), 3)
            return new_image
    return None

def lineNotify(token, msg):
    url ='https://notify-api.line.me/api/notify'
    headers = {
        "Authorization":"Bearer " + token,
        "Content-Type":"application/x-www-form-urlencoded"
    }
    payload = {'message': msg}
    r = requests.post(url,headers=headers,params=payload)
    return r.status_code

if __name__ == '__main__':
    camera_open = False

    #LINE token
    token = 'qey0fmUdlbaOXiwFTttS4YXCc1GJdwMsUZUuXuAUJNk'

    print('※ 開始執行YOLOV4物件偵測...')
    model = initNet()
    if camera_open:
        cap = cv2.VideoCapture(0)
    while(True):
        if camera_open:
            ret , img =cap.read()
        else:
            image_path = r'VOCdevkit\VOC2021\JPEGImages\007.jpg'
            img = cv2.imread(image_path)
        classes, confs, boxes = nnProcess(img, model)
        if len(boxes) == 0:
            continue
        box_img=read_class(img,classes,confs,boxes,1)
        img=cut_img(img,classes,confs,boxes)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # clahe = cv2.createCLAHE()
        # img_gray = clahe.apply(img_gray)
        kernel = np.ones((3,3),np.uint8)
        # dilate = cv2.dilate(img_gray,kernel,iterations=1)
        erode = cv2.erode(img_gray,kernel,iterations=1)

        ret, thresh = cv2.threshold(erode, 100, 255, cv2.THRESH_BINARY_INV)
        count ,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        copy_img = img.copy()
        for cnt in count:
            if len(cnt) > 0:
                # print(len(cnt))
                if cv2.contourArea(cnt) < 5000:
                    continue
                # print(cv2.contourArea(cnt))
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(copy_img,(x,y),(x+w,y+h),(0,255,0),5)
                crop_img = img[y:y+h,x:x+w]
                # cv2.imshow('main3',crop_img)
                # convert to grayscale
                level_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                level_img, img_bin = cv2.threshold(level_img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                level_img = cv2.bitwise_not(img_bin)
                kernel = np.ones((3, 3), np.uint8)

                # make the image bigger, because it needs at least 30 pixels for the height for the characters
                level_img = cv2.resize(level_img,(0,0),fx=4,fy=4, interpolation=cv2.INTER_CUBIC)
                level_img = cv2.dilate(level_img, kernel, iterations=1)

                # --debug--
                cv2.imshow("Debug", level_img)
                config = r'-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3'
                level = pytesseract.image_to_string(level_img, config=config)
                print(level)
        with open('output.csv','a+') as f:
            f.write(level.replace('\n',''))
            f.write(',')
        lineTool.lineNotify(token,'OCR:'+level)
        cv2.imshow('main',img)
        # cv2.imshow('main2',thresh)
        # cv2.imshow('main4',copy_img)
        
        if camera_open == False:
            cv2.waitKey(0)
            break
        else:
            cv2.waitKey(10)
cv2.destroyAllWindows()
