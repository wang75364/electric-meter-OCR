import requests
import cv2
from cv2 import aruco

#pip install opencv-python==4.5.5.62
#pip install opencv-contrib-python==4.5.5.62

if __name__ == '__main__':
    #API 網址
    url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/F-D0047-093?Authorization=CWB-3C9AC77F-E9B0-466D-A0B8-0065FDFBC8EE&locationId=F-D0047-025'
    #圖片路徑
    img = cv2.imread('singlemarkersoriginal.jpg')
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
                print(f'{city}\n12小時降雨機率:{PoP12h}%\n天氣現象: {Wx}\n體感溫度:{AT} ℃\n溫度:{T}℃\n相對濕度:{RH}%\n舒適度指數:{CI}NA {CI_str}\n天氣預報綜合描述:{WeatherDescription}\n6小時降雨機率:{PoP6h}%\n')
        
        # show the output image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
            
cv2.destroyAllWindows()