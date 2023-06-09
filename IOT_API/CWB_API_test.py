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
    #API 網址
    url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/F-D0047-093?Authorization=CWB-3C9AC77F-E9B0-466D-A0B8-0065FDFBC8EE&locationId=F-D0047-025'
    #LINE token
    token = 'line token'

    print('※ 開始執行YOLOV4物件偵測...')
    model = initNet()
    #圖片路徑
    image_path = r'C:\UserProgram\python_test\IOT_API\VOCdevkit\VOC2021\JPEGImages\007.jpg'
    img = cv2.imread(image_path)
    classes, confs, boxes = nnProcess(img, model)
    box_img=read_class(img,classes,confs,boxes,1)
    img=cut_img(img,classes,confs,boxes)

    img = cv2.resize(img,[2560,1280])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE()
    img_gray = clahe.apply(img_gray)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(img_gray,kernel,iterations=2)
    erode = cv2.erode(dilate,kernel,iterations=1)

    ret, thresh = cv2.threshold(erode, 200, 255, cv2.THRESH_BINARY_INV)
    count ,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    copy_img = img.copy()
    for cnt in count:
        if len(cnt) > 0:
            # print(len(cnt))
            # print(cv2.contourArea(cnt))
            if cv2.contourArea(cnt) < 500000:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            # print(x,y,w,h)
            cv2.rectangle(copy_img,(x,y),(x+w,y+h),(0,255,0),5)
            # x,y,w,h = cv2.boundingRect(cnt)
            crop_img = thresh[y+100:y+h-100,x+100:x+w-100]
    crop_img = cv2.resize(crop_img,[400,200])
    config = r'-c tessedit_char_whitelist=0123456789 --psm 6'
    ocr_rst=pytesseract.image_to_string(crop_img,  config=config,lang="eng")
    # print(ocr_rst)
    with open('output.csv','a+') as f:
        f.write(ocr_rst)
        f.write(',')
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
            print(f'{city}\n12小時降雨機率:{PoP12h}%\n天氣現象: {Wx}\n體感溫度:{AT} ℃\n溫度:{T}℃\n相對濕度:{RH}%\n舒適度指數:{CI}NA {CI_str}\n天氣預報綜合描述:{WeatherDescription}\n6小時降雨機率:{PoP6h}%\n')
    
    lineTool.lineNotify(token,WeatherDescription)
    lineTool.lineNotify(token,'OCR:'+ocr_rst)
    cv2.imshow('main2',crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
