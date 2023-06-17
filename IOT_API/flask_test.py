from flask import Flask, render_template, Response,request
import cv2
from cv2 import aruco
import lineTool
import requests
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


app = Flask(__name__)
camera = cv2.VideoCapture(0)
frame =None
camera_open = True
#LINE token
token = 'qey0fmUdlbaOXiwFTttS4YXCc1GJdwMsUZUuXuAUJNk'

print('※ 開始執行YOLOV4物件偵測...')
model = initNet()

def gen_frames():
    while True:
        global frame
        success, frame = camera.read()
        # success = True
        # frame = cv2.imread(r'C:\UserProgram\python_test\IOT_API\VOCdevkit\VOC2021\JPEGImages\002.jpg')
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            buffer = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rtmain', methods=['GET','POST'])
def rtmain():
    return render_template('index.html')

@app.route("/test", methods=['GET','POST'])
def test():
    if request.method == 'POST':
        # print("Request.method:", request.method)
        # print(request.form.get('action'))
        if 'Aruco' in request.form.get('action'):
            if type(frame) == type(None):
                return render_template('index.html')
            url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/F-D0047-093?Authorization=CWB-3C9AC77F-E9B0-466D-A0B8-0065FDFBC8EE&locationId=F-D0047-025'
            #圖片路徑
            # img = cv2.imread('singlemarkersoriginal.jpg')
            img = frame
            arucoDict = aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            arucoParams = aruco.DetectorParameters_create()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,parameters=arucoParams)
            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned in
                    # top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                # draw the bounding box of the ArUCo detection
                cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
                cv2.putText(img, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[INFO] ArUco marker ID: {}".format(markerID))

                data = requests.get(url)   # 取得 JSON 檔案的內容為文字
                data_json = data.json()    # 轉換成 JSON 格式
                location = data_json['records']['locations'][0]['location']
                for i in location:
                    city = i['locationName']    # 縣市名稱
                    if "虎尾" in city:
                        PoP12h = i['weatherElement'][0]['time'][0]['elementValue'][0]['value']    # 12小時降雨機率
                        Wx = i['weatherElement'][1]['time'][0]['elementValue'][0]['value']        # 天氣現象
                        AT = i['weatherElement'][2]['time'][0]['elementValue'][0]['value']    # 體感溫度
                        T = i['weatherElement'][3]['time'][0]['elementValue'][0]['value']    # 溫度
                        RH = i['weatherElement'][4]['time'][0]['elementValue'][0]['value']    # 相對濕度
                        CI = i['weatherElement'][5]['time'][0]['elementValue'][0]['value']    # 舒適度指數
                        CI_str = i['weatherElement'][5]['time'][0]['elementValue'][1]['value']    # 舒適度指數
                        WeatherDescription = i['weatherElement'][6]['time'][0]['elementValue'][0]['value']    # 天氣預報綜合描述
                        PoP6h = i['weatherElement'][7]['time'][0]['elementValue'][0]['value']    # 6小時降雨機率
                        WS = i['weatherElement'][8]['time'][0]['elementValue'][0]['value']    # 風速
                        WD = i['weatherElement'][9]['time'][0]['elementValue'][0]['value']    # 風向
                        Td = i['weatherElement'][10]['time'][0]['elementValue'][0]['value']    # 露點溫度
                        text_tr = "[INFO] ArUco marker ID: {}\n".format(markerID) + f'{city}\n12小時降雨機率:{PoP12h}%\n天氣現象: {Wx}\n體感溫度:{AT} ℃\n溫度:{T}℃\n相對濕度:{RH}%\n舒適度指數:{CI}NA {CI_str}\n天氣預報綜合描述:{WeatherDescription}\n6小時降雨機率:{PoP6h}%\n'
                        print(text_tr)
                        return render_template('index2.html',sample_input=text_tr)
        elif 'OCR' in request.form.get('action'):
            if type(frame) != type(None):
                # print(type(frame))
                classes, confs, boxes = nnProcess(frame, model)
                if len(boxes) != 0:
                    # box_img=read_class(frame,classes,confs,boxes,1)
                    img=cut_img(frame,classes,confs,boxes)
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
                            # convert to grayscale
                            level_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            level_img, img_bin = cv2.threshold(level_img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            level_img = cv2.bitwise_not(img_bin)
                            kernel = np.ones((3, 3), np.uint8)

                            # make the image bigger, because it needs at least 30 pixels for the height for the characters
                            level_img = cv2.resize(level_img,(0,0),fx=4,fy=4, interpolation=cv2.INTER_CUBIC)
                            level_img = cv2.dilate(level_img, kernel, iterations=1)

                            config = r'-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3'
                            level = pytesseract.image_to_string(level_img, config=config)
                            print(level)
                            with open('output.csv','a+') as f:
                                f.write(level.replace('\n',''))
                                f.write(',')
                            lineTool.lineNotify(token,'OCR:'+level)
        else:
            print('0003C')    
    elif request.method == 'GET':
        print("Request.method:", request.method)
    return render_template('index.html')


if __name__ == '__main__':
    app.run('0.0.0.0')