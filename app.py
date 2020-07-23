from flask import Flask
from flask_restful import Resource, Api, reqparse
import MySQLdb
import json
import jaconv
from difflib import get_close_matches
from flask_cors import CORS
from connection import *
from const import *

DEBUG_MODE = False
DEBUG_WITH_IMAGE = False

app = Flask(__name__)
api = Api(app)

# Create connection
db = MySQLdb.connect(host_name,
                     user_name,
                     user_password,
                     database,
                     charset=options["char"])


def box_weighted(dataBox):
    return dataBox[0].vertices[0].y


def box_weighted_x(dataBox):
    return dataBox[0].vertices[0].x


def get_mean_height(dataBoxes, rotation):
    sum = 0
    count = 0
    for databox in dataBoxes:
        if len(databox) == 3:
            try:
                if databox[2] == ('horizontal' if rotation else 'vertical'):
                    count += 1
                    sum += abs(databox[0].vertices[0].y - databox[0].vertices[3].y) if rotation else abs(
                        databox[0].vertices[0].x - databox[0].vertices[3].x)
            except:
                continue
    if count == 0:
        return 0
    return sum / count


def get_max_height(dataBoxes):
    max = 0
    for databox in dataBoxes:
        try:
            if len(databox) == 3:
                if databox[2] == 'horizontal':
                    if max < abs(databox[0].vertices[0].y - databox[0].vertices[3].y):
                        max = abs(databox[0].vertices[0].y - databox[0].vertices[3].y)
        except:
            continue
    return max


def image_resize(image, width=None, height=None, inter=None):
    import cv2

    if inter is None:
        inter = cv2.INTER_AREA

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)

    else:

        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def crop_advance(image, points):
    minX = image.shape[1]
    maxX = -1
    minY = image.shape[0]
    # maxY = -1
    # for point in polygon[0]:
    #
    #     x = point[0]
    #     y = point[1]
    #
    #     if x < minX:
    #         minX = x
    #     if x > maxX:
    #         maxX = x
    #     if y < minY:
    #         minY = y
    #     if y > maxY:
    #         maxY = y
    #
    # # Go over the points in the image if thay are out side of the emclosing rectangle put zero
    # # if not check if thay are inside the polygon or not
    # cropedImage = np.zeros_like(I)
    # for y in range(0, I.shape[0]):
    #     for x in range(0, I.shape[1]):
    #
    #         if x < minX or x > maxX or y < minY or y > maxY:
    #             continue
    #
    #         if cv2.pointPolygonTest(np.asarray(polygon), (x, y), False) >= 0:
    #             cropedImage[y, x, 0] = I[y, x, 0]
    #             cropedImage[y, x, 1] = I[y, x, 1]
    #             cropedImage[y, x, 2] = I[y, x, 2]
    #
    # # Now we can crop again just the envloping rectangle
    # finalImage = cropedImage[minY:maxY, minX:maxX]


def crop_img(image, x1, y1, x2, y2):
    height, width = image.shape

    if x1 < 0:
        x1 = 0
    if x2 > width:
        x2 = width

    return image[y1:y2, x1:x2]


def request_api(image, text_rows, pos):
    from google.cloud import vision
    import cv2

    client = vision.ImageAnnotatorClient()

    success, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()

    image = vision.types.Image(content=content)

    try:
        response = client.text_detection(image=image,
                                         # image_context={"language_hints": ["ja"]}, #for some reason this hint make it
                                         # worse
                                         )
        content = response.text_annotations[0].description
    except:
        if DEBUG_MODE:
            cv2.imshow('Test', image)
            cv2.waitKey(0)
    text_rows[pos] = content.split()
    # print(content.split(), flush=True)


def OCR_Extract_relative(equalized, dataBoxes, rotation):
    import cv2

    dataBoxes.sort(key=box_weighted if rotation else box_weighted_x)
    # print(dataBoxes)

    # print(resText)
    rows = []
    row = []
    pre = -1000
    mean = get_mean_height(dataBoxes, rotation) / 2
    debug_img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    count = 0
    for dataBox in dataBoxes:
        if count == 0:
            count += 1
            continue
        try:
            if dataBox[2] == ('vertical' if rotation else 'horizontal'):
                count += 1
                continue
        except:
            continue

        if pre == -1000 or abs((dataBox[0].vertices[0].y if rotation else dataBox[0].vertices[0].x) - pre) <= mean:
            pre = dataBox[0].vertices[0].y if rotation else dataBox[0].vertices[0].x
            row.append(dataBox.copy())

        else:
            rows.append(row.copy())
            row = [dataBox.copy()]
            pre = -1000

        count += 1
    rows.append(row.copy())

    for row in rows:
        row.sort(key=box_weighted_x if rotation else box_weighted)

    if DEBUG_WITH_IMAGE:
        from random import randint
        for row in rows:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            count = 0
            for dataBox in row:
                rect = dataBox[0]
                count += 1
                cv2.putText(debug_img, str(count), (rect.vertices[0].x, rect.vertices[0].y - 10), cv2.FONT_ITALIC, 1,
                            color, 2)
                cv2.line(debug_img, (rect.vertices[0].x, rect.vertices[0].y), (rect.vertices[1].x, rect.vertices[1].y),
                         color, thickness=2)
                cv2.line(debug_img, (rect.vertices[1].x, rect.vertices[1].y), (rect.vertices[2].x, rect.vertices[2].y),
                         color, thickness=2)
                cv2.line(debug_img, (rect.vertices[2].x, rect.vertices[2].y), (rect.vertices[3].x, rect.vertices[3].y),
                         color, thickness=2)
                cv2.line(debug_img, (rect.vertices[3].x, rect.vertices[3].y), (rect.vertices[0].x, rect.vertices[0].y),
                         color, thickness=2)

                cv2.circle(debug_img, (rect.vertices[0].x, rect.vertices[0].y), 2, (0, 255, 0), 2)
                cv2.circle(debug_img, (rect.vertices[1].x, rect.vertices[1].y), 2, (255, 0, 0), 2)
                cv2.circle(debug_img, (rect.vertices[2].x, rect.vertices[2].y), 2, (0, 0, 255), 2)
                cv2.circle(debug_img, (rect.vertices[3].x, rect.vertices[3].y), 2, (255, 255, 0), 2)
                cv2.imshow('TEST', debug_img)
        cv2.waitKey(0)

    return rows


def searchRowForLast(rows, characters):
    pos = []
    row_count = -1
    for row in rows:
        row_count += 1
        item_count = -1
        for item in row:
            item_count += 1
            if any(kanji_char in item[1] for kanji_char in characters):
                pos = [row_count, item_count]
    if DEBUG_MODE:
        print('first pos=', pos)
    if len(pos) == 0:
        return None
    if len(rows[pos[0]]) - 1 > pos[1]:
        return [pos[0], pos[1] + 1]
    else:
        return [pos[0] + 1, 0]


def searchRowForFirst(rows, characters):
    pos = []
    row_count = -1
    for row in rows:
        row_count += 1
        item_count = -1
        for item in row:
            item_count += 1
            if any(kanji_char in item[1] for kanji_char in characters):
                pos = [row_count, item_count]
                print('last pos=', pos)
                if len(rows[pos[0]]) - 1 > pos[1]:
                    return [pos[0], pos[1] + 1]
                else:
                    return [pos[0] + 1, 0]


def OCR_Field_Up(typeLabel, rows, rotation, dataBoxes):
    if typeLabel is None:
        return None

    import re

    if DEBUG_MODE:
        for row in rows:
            for item in row:
                print(item[1], end=',')
            print()

    res = {}
    if typeLabel <= 1:

        pos = searchRowForLast(rows, ['商', '品', '名'])
        res["zaiHinname"] = ''
        for i in range(pos[1], len(rows[pos[0]])):
            res["zaiHinname"] += rows[pos[0]][i][1]

        pos = searchRowForLast(rows, ['種', '類'])
        res["zaiKikaku"] = rows[pos[0]][pos[1]][1]

        pos = searchRowForLast(rows, ['断', '面', '寸', '法'])
        res['zaiDansun'] = rows[pos[0]][pos[1]][1]

        tmpZaiString = ''
        pos1 = searchRowForLast(rows, ['長', 'さ'])
        pos2 = searchRowForFirst(rows, ['本', '数'])
        for i in range(pos1[1], pos2[1] - 1):
            tmpZaiString += rows[pos1[0]][i][1]
        zailength_str = re.sub("[^0-9.]", "", tmpZaiString)
        try:
            res['zaiLength'] = float(zailength_str)
        except:
            res['zaiLength'] = float(-1)
        res['zaiLengthUnit'] = tmpZaiString.replace(zailength_str, '')

        pos = searchRowForLast(rows, ['本', '数'])
        zaiInzu_str = rows[pos[0]][pos[1]][1]
        try:
            res['zaiInzu'] = int(zaiInzu_str)
        except:
            res['zaiInzu'] = -1

        pos = searchRowForLast(rows, ['年月'])
        res["zaiSozaiSeizobi"] = ''
        for i in range(pos[1], len(rows[pos[0]])):
            res["zaiSozaiSeizobi"] += rows[pos[0]][i][1]

        pos = searchRowForFirst(rows, ['番', '号'])
        res["zaiSozaino"] = rows[pos[0]][pos[1]][1]

        # dung quen buoc tien xu ly Sort o cho nay nbhe!!!!
        if typeLabel == 0:
            res['zaiCharge'] = ''
            first = True
            for databox in dataBoxes:
                if first:
                    first = False
                    continue
                try:
                    if databox[2] == ('vertical' if rotation else 'horizontal'):
                        res['zaiCharge'] += databox[1]
                except:
                    continue
        else:
            res['zaiCharge'] = ''
            pos = searchRowForLast(rows, ['鉄', '建', '材'])
            for i in range(0, len(rows[pos[0] - 1])):
                res["zaiCharge"] += rows[pos[0] - 1][i][1]

    if typeLabel == 2:
        tmp_zaiHinname = ''
        for item in rows[0]:
            tmp_zaiHinname += item[1]
        res["zaiHinname"] = tmp_zaiHinname

        tmp_zaiKikaku = ''
        maxLen = 0
        for item in rows[1]:
            if maxLen < len(item[1]):
                tmp_zaiKikaku = item[1]
                maxLen = len(item[1])
        res["zaiKikaku"] = tmp_zaiKikaku

        tmpzaiDansun = ''

        for item in rows[1]:
            if "." in item[1]:
                tmpzaiDansun += "x" + item[1]
                break

        for item in rows[2]:
            if "X" in item[1] or "x" in item[1]:
                tmpzaiDansun = item[1] + tmpzaiDansun
                break
        res['zaiDansun'] = tmpzaiDansun

        if len(rows[3]) > 1:
            tmpZaiStr = ''
            for item in rows[3]:
                if "." in item[1]:
                    tmpZaiStr = item[1]
                    break
            tmpZaiLength = re.sub("[^0-9.]", "", tmpZaiStr)
            try:
                res['zaiLength'] = float(tmpZaiLength)
            except:
                res['zaiLength'] = float(-1)

            res['zaiLengthUnit'] = tmpZaiStr.replace(tmpZaiLength, '')

            for item in rows[3]:
                if ":" in item[1] or len(re.sub("[^0-9.]", "", item[1])) == len(item[1]):
                    try:
                        res['zaiInzu'] = int(item[1].replace(':', ''))
                        break
                    except:
                        continue
        else:
            res['zaiLength'] = ''
            res['zaiLengthUnit'] = ''
            res['zaiInzu'] = -1

        res['zaiSozaino'] = ''
        for item in rows[5]:
            res['zaiSozaino'] += item[1]

        tmp_zaiCharge = ''
        maxLen = 0
        for item in rows[7]:
            if maxLen < len(item[1]):
                tmp_zaiCharge = item[1]
                maxLen = len(item[1])
        res["zaiCharge"] = tmp_zaiCharge

    #

    return res


def OCR_row_extract(rows, equalized, labelType, dataBoxes, rotation):
    import cv2
    from threading import Thread
    from random import randint
    import re

    threads = []
    text_rows = len(rows) * ['']
    pos = -1

    for row in rows:
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        lowest_bottom_y = -1000000
        highest_top_y = 1000000

        for rect in row:
            highest_top_y = min(highest_top_y, rect[0].vertices[0].y, rect[0].vertices[1].y)
            lowest_bottom_y = max(lowest_bottom_y, rect[0].vertices[2].y, rect[0].vertices[3].y)

        pos += 1
        l = row[-1][0].vertices[2].x - row[0][0].vertices[0].x
        currRowCrop = crop_img(equalized, int(row[0][0].vertices[0].x - l * 0.25), highest_top_y - 5,
                               int(row[-1][0].vertices[2].x + l * 0.7), lowest_bottom_y + 5)

        session_thread = Thread(target=request_api, args=(currRowCrop, text_rows, pos,))
        session_thread.start()

        threads.append(session_thread)
        if DEBUG_WITH_IMAGE:
            cv2.imshow('TEST', currRowCrop)
            cv2.waitKey(0)

    for thread in threads:
        thread.join()

    if DEBUG_MODE:
        for texts in text_rows:
            print(texts)
    res = {}
    if labelType <= 1:
        zaihin_name_str = ''
        pos = search_for_keywords(text_rows[0], ['商', '品', '名'])
        for i in range(pos + 1, len(text_rows[0])):
            zaihin_name_str += text_rows[0][i]
        res['zaiHinname'] = zaihin_name_str

        res['zaiKikaku'] = ''
        pos = search_for_keywords(text_rows[1], ['種', '類'])
        for i in range(pos + 1, len(text_rows[1])):
            res['zaiKikaku'] += text_rows[1][i]
        res['zaiKikaku'] = re.sub("[^0-9a-zA-Z]", "", res['zaiKikaku'])

        res['zaiDansun'] = ''
        pos = search_for_keywords(text_rows[2], ['断', '面', '寸', '法'])
        for i in range(pos + 1, len(text_rows[2])):
            res['zaiDansun'] += text_rows[2][i]
        if len(text_rows[2]) == 1:
            res['zaiDansun'] = text_rows[2][0]
        res['zaiDansun'] = re.sub("[^0-9x]", "", res['zaiDansun'])

        pos1 = search_for_keywords(text_rows[3], ['長', 'さ'])
        pos2 = search_for_keywords(text_rows[3], ['本', '数'])

        if (pos2 - pos1 >= 3):
            try:
                res['zaiLength'] = float(text_rows[3][pos1 + 1])
            except:
                res['zaiLength'] = -1
            res['zaiLengthUnit'] = text_rows[3][pos2 - 1]

        else:
            if (pos2 - pos1 == 1):
                zai_arr = text_rows[3][pos2].split('本数')
                zai_length_str = re.sub("[^0-9.]", "", zai_arr[0])
                try:
                    res['zaiLength'] = float(zai_length_str)
                except:
                    res['zaiLength'] = -1
                res['zaiLengthUnit'] = zai_arr[0].replace(zai_length_str, '')
                try:
                    res['zaiInzu'] = int(zai_arr[1])
                except:
                    res['zaiInzu'] = -1
            else:
                zai_length_str = re.sub("[^0-9.]", "", text_rows[3][pos1 + 1])
                try:
                    res['zaiLength'] = float(zai_length_str)
                except:
                    res['zaiLength'] = -1
                res['zaiLengthUnit'] = text_rows[3][pos1 + 1].replace(zai_length_str, '')

        if 'zaiInzu' not in res:
            try:
                if pos2 + 1 == len(text_rows[3]):
                    res['zaiInzu'] = int(re.sub("[^0-9]", "", text_rows[3][pos2]))
                else:
                    res['zaiInzu'] = int(re.sub("[^0-9]", "", text_rows[3][pos2 + 1]))
            except:
                res['zaiInzu'] = -1

        if res['zaiInzu'] == -1:
            pos = searchRowForLast(rows, ['本', '数'])
            zaiInzu_str = rows[pos[0]][pos[1]][1]
            try:
                res['zaiInzu'] = int(zaiInzu_str)
            except:
                res['zaiInzu'] = -1

        pos = searchRowForLast(rows, ['年月'])
        res["zaiSozaiSeizobi"] = ''
        for i in range(pos[1], len(rows[pos[0]])):
            res["zaiSozaiSeizobi"] += rows[pos[0]][i][1]

        pos = search_for_keywords(text_rows[5], ['番', '号'])
        res["zaiSozaino"] = ''
        for i in range(pos + 1, len(text_rows[5])):
            res["zaiSozaino"] += text_rows[5][i]

        if len(res['zaiSozaino']) == 0:
            pos = searchRowForFirst(rows, ['番', '号'])
            res["zaiSozaino"] = rows[pos[0]][pos[1]][1]

        # dung quen buoc tien xu ly Sort o cho nay nbhe!!!!
        if labelType == 0:
            res['zaiCharge'] = ''
            first = True
            for databox in dataBoxes:
                if first:
                    first = False
                    continue
                try:
                    if databox[2] == ('vertical' if rotation else 'horizontal'):
                        res['zaiCharge'] += databox[1]
                except:
                    continue
        else:
            res['zaiCharge'] = ''
            pos = searchRowForLast(rows, ['鉄', '建', '材'])
            for i in range(0, len(rows[pos[0] - 1])):
                res["zaiCharge"] += rows[pos[0] - 1][i][1]

    if labelType == 2:
        tmp_zaiHinname = ''
        for item in rows[0]:
            tmp_zaiHinname += item[1]
        res["zaiHinname"] = tmp_zaiHinname

        tmp_zaiKikaku = ''
        maxLen = 0
        for item in rows[1]:
            if maxLen < len(item[1]):
                tmp_zaiKikaku = item[1]
                maxLen = len(item[1])
        res["zaiKikaku"] = tmp_zaiKikaku

        tmpzaiDansun = ''

        for item in rows[1]:
            if "." in item[1]:
                tmpzaiDansun += "x" + item[1]
                break

        for item in rows[2]:
            if "X" in item[1] or "x" in item[1]:
                tmpzaiDansun = item[1] + tmpzaiDansun
                break
        res['zaiDansun'] = tmpzaiDansun

        if len(rows[3]) > 1:
            tmpZaiStr = ''
            for item in rows[3]:
                if "." in item[1]:
                    tmpZaiStr = item[1]
                    break
            tmpZaiLength = re.sub("[^0-9.]", "", tmpZaiStr)
            try:
                res['zaiLength'] = float(tmpZaiLength)
            except:
                res['zaiLength'] = float(-1)

            res['zaiLengthUnit'] = tmpZaiStr.replace(tmpZaiLength, '')

            for item in rows[3]:
                if ":" in item[1] or len(re.sub("[^0-9.]", "", item[1])) == len(item[1]):
                    try:
                        res['zaiInzu'] = int(item[1].replace(':', ''))
                        break
                    except:
                        continue
        else:
            res['zaiLength'] = ''
            res['zaiLengthUnit'] = ''
            res['zaiInzu'] = -1

        res['zaiSozaino'] = ''
        for item in rows[5]:
            res['zaiSozaino'] += item[1]

        tmp_zaiCharge = ''
        maxLen = 0
        for item in rows[7]:
            if maxLen < len(item[1]):
                tmp_zaiCharge = item[1]
                maxLen = len(item[1])
        res["zaiCharge"] = tmp_zaiCharge

    return res


def search_for_keywords(row, characters):
    item_count = -1
    for item in row:
        item_count += 1
        if any(kanji_char in item for kanji_char in characters):
            return item_count
    return -1


def process_data(inputCVImg):
    import numpy as np
    import cv2
    from google.cloud import vision

    gray = cv2.cvtColor(inputCVImg, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(equalized)

    # equalized = image_resize(equalized,height=400)

    # equalized = cv2.rotate(equalized, rotateCode=cv2.ROTATE_90_CLOCKWISE)

    # ret, equalized = cv2.threshold(equalized,140,255,cv2.THRESH_BINARY)
    #
    # cv2.imshow('haha',equalized)
    # cv2.waitKey(0)

    client = vision.ImageAnnotatorClient()

    uri = ''

    success, encoded_image = cv2.imencode('.jpg', equalized)
    content = encoded_image.tobytes()

    if len(uri) == 0:
        image = vision.types.Image(content=content)
    else:
        image = vision.types.Image()
        image.source.image_uri = uri

    response = client.text_detection(image=image,
                                     # image_context={"language_hints": ["ja"]}, #for some reason this hint make it
                                     # worse
                                     )
    # print(response)
    dataBoxes = []

    texts = response.text_annotations
    for text in texts:
        dataBoxes.append([text.bounding_poly, text.description])

    count_vertical_bound = 0
    count_horizontal_bound = 0
    first = True
    for dataBox in dataBoxes:
        if first:
            first = False
            continue
        rect = dataBox[0]
        if abs(rect.vertices[0].x - rect.vertices[1].x) > abs(rect.vertices[0].y - rect.vertices[1].y):
            count_horizontal_bound += 1
            dataBox.append('horizontal')
        else:
            if abs(rect.vertices[0].x - rect.vertices[1].x) < abs(rect.vertices[0].y - rect.vertices[1].y):
                count_vertical_bound += 1
                dataBox.append('vertical')
    if count_horizontal_bound > 0 and count_vertical_bound > 0:
        print('Nikken shiga NGANG!')
        typeLabel = 0
    else:
        if '商品名' in dataBoxes[0][1]:
            print('Nikken shiga DOC!')
            typeLabel = 1
        else:
            print('NSMP!')
            typeLabel = 2
    if count_vertical_bound > count_horizontal_bound:
        print('sai chieu')
        rotation = False
    else:
        print('dung chieu')
        rotation = True

    res = None

    rows = OCR_Extract_relative(equalized, dataBoxes, rotation)
    # try:
    #     print(OCR_Field_Up(typeLabel, rows, rotation, dataBoxes))
    # except:
    #     print("Error")
    # res = OCR_row_extract(rows, equalized,typeLabel,dataBoxes,rotation)

    try:
        res = OCR_Field_Up(typeLabel, rows, rotation, dataBoxes)
    except:
        print("Error")

    return res


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def find_last(s, first):
    try:
        start = s.index(first) + len(first)
        end = len(s) - 1
        return s[start:end]
    except ValueError:
        return ""


class OCR_API(Resource):
    @staticmethod
    def get():
        return {'about': 'OCR api'}

    @staticmethod
    def post():
        import werkzeug
        import numpy as np
        import cv2 as cv

        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        media_file = args['file']

        npImg = np.fromstring(media_file.read(), np.uint8)
        try:
            cvImg = cv.imdecode(npImg, cv.IMREAD_UNCHANGED)
        except:
            return None
        # barcode_detect(cvImg)
        res = process_data(cvImg)
        print("RES_OCR: ", res)
        return res


api.add_resource(OCR_API, '/ocr_api/hello')


class HandleRequest(Resource):

    def __init__(self):
        self.zai_length_unit = ""
        self.zai_hinname = ""
        self.zai_kikaku = ""
        self.zai_dansun = ""
        self.zai_length = ""
        self.zai_inzu = ""
        self.zai_sozaiseizobi = ""
        self.zai_sozaino = ""
        self.zai_charge = ""

    @staticmethod
    def read_json_file():
        with open('img_content.json', encoding='utf-8') as json_file:
            result = json.load(json_file)
            return result

    def handle_data_receiver(self):
        data = self.read_json_file()
        print("DATA: ", data)
        return data

    def get_hinname_from_database(self):
        try:
            cursor = db.cursor()
            sql = 'SELECT * FROM morisada.m_kihon_tanju where kikaku = %s and dansun_a = %s and dansun_b = %s and dansun_t = %s'
            kikaku = self.zai_kikaku
            dansun = self.zai_dansun
            arr_dansun = dansun.split("x")
            values = (kikaku, arr_dansun[0], arr_dansun[1], arr_dansun[2])
            cursor.execute(sql, values)
            records = cursor.fetchall()
            if len(records) != 0:
                for record in records:
                    return record[2]
            else:
                return self.zai_hinname
        except (MySQLdb.OperationalError, IndexError):
            return self.zai_hinname

    def get_hinname(self, *args):
        print("ALO: ", get_close_matches(self.zai_hinname, args[1]))
        arr_found_hinnname = get_close_matches(self.zai_hinname, args[1])
        if len(arr_found_hinnname) == 0:
            return ""
        found_354 = args[0].find('354')
        if (len(self.zai_hinname) == 7) and (not found_354 == -1):
            old_string = self.zai_hinname
            new_string = old_string.replace("354", "ｺﾗﾑ")
            return new_string
        else:
            most_similar_string_zai_hinname = get_close_matches(args[0], args[1])
            if len(most_similar_string_zai_hinname) == 0:
                return self.get_hinname_from_database()
            else:
                return most_similar_string_zai_hinname[0]

    def reset_value(self, **kwargs):
        result = []
        dansun_empty = kwargs["dansun_empty"]
        length_obj = kwargs["len"]
        arr_kikaku = kwargs["arr_value"]["arr_kikaku"]
        arr_hinname = kwargs["arr_value"]["arr_hinname"]
        a = kwargs["dansun_obj"]["dansun_a"]
        b = kwargs["dansun_obj"]["dansun_b"]
        t = kwargs["dansun_obj"]["dansun_t"]
        data_kikaku = get_close_matches(self.zai_kikaku, arr_kikaku)
        hinname = self.get_hinname(self.zai_hinname, arr_hinname)
        if dansun_empty:
            for i in range(0, length_obj):
                if (self.zai_kikaku == arr_kikaku[i]) and (hinname == arr_hinname[i]):
                    new_dansun = str(int(a[i])) + "x" + str(int(b[i])) + "x" + str(int(t[i]))
                    result.append(new_dansun)
            if len(result) == 0:
                return ""
            else:
                return result[0]
        else:
            if not self.zai_hinname:
                for i in range(0, length_obj):
                    new_dansun = str(int(a[i])) + "x" + str(int(b[i])) + "x" + str(int(t[i]))
                    if (self.zai_kikaku == arr_kikaku[i]) and (self.zai_dansun == new_dansun):
                        result.append(arr_hinname[i])
                if len(result) == 0:
                    return ""
                else:
                    return result[0]
            elif not self.zai_kikaku:
                for i in range(0, length_obj):
                    new_dansun = str(int(a[i])) + "x" + str(int(b[i])) + "x" + str(int(t[i]))
                    if (self.zai_hinname == arr_hinname[i]) and (self.zai_dansun == new_dansun):
                        result.append(arr_kikaku[i])
                if len(result) == 0:
                    return ""
                else:
                    print(result)
                    if len(result) >= 1:
                        return ""
                    else:
                        return result[0]

    def handle_zai_sozaino(self):
        sozaino = self.zai_sozaino
        a_list = list(range(0, 10))
        list_ref = [str(x) for x in a_list]
        if len(sozaino) == 3:
            if type(sozaino[2]) != "number":
                if sozaino[2] == "S":
                    sozaino = sozaino[0:2] + sozaino[2:3].replace(sozaino[2:3], "5")
                if sozaino[2] == "O":
                    sozaino = sozaino[0:2] + sozaino[2:3].replace(sozaino[2:3], "0")

            if sozaino[0] in list_ref:
                if sozaino[0] == "0":
                    sozaino = sozaino[0:1].replace(sozaino[0:1], "O") + sozaino[1:]
                if sozaino[0] == "2":
                    sozaino = sozaino[0:1].replace(sozaino[0:1], "Z") + sozaino[1:]

        return sozaino

    def post(self):
        import werkzeug
        import numpy as np
        import cv2 as cv

        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        media_file = args['file']

        npImg = np.fromstring(media_file.read(), np.uint8)
        try:
            cvImg = cv.imdecode(npImg, cv.IMREAD_UNCHANGED)
        except:
            return None
        # barcode_detect(cvImg)

        # Test for get response from api
        data = process_data(cvImg)

        # Test for read data from file
        # data = self.handle_data_receiver()
        new_response = data

        # Alter request

        self.zai_length_unit = data[zaiLengthUnit]
        self.zai_hinname = jaconv.z2h(data[zaiHinname].format())
        self.zai_kikaku = data[zaiKikaku]
        self.zai_dansun = data[zaiDansun]
        self.zai_length = data[zaiLength]
        self.zai_inzu = data[zaiInzu]
        self.zai_sozaiseizobi = data[zaiSozaiSeizobi]
        self.zai_sozaino = data[zaiSozaino]
        self.zai_charge = data[zaiCharge]

        arr_header = []

        # create 7 empty list
        kihon_tanju_cd, kikaku, hin_name, dansun_t, dansun_a, dansun_b, keisan_hoshiki = ([] for i in range(7))
        try:
            cursor = db.cursor()
            sql = 'select * from m_kihon_tanju'
            cursor.execute(sql)
            records = cursor.fetchall()
            for col in records:
                kihon_tanju_cd.append(col[0])
                kikaku.append(col[1])
                hin_name.append(col[2])
                dansun_t.append(col[3])
                dansun_a.append(col[4])
                dansun_b.append(col[5])
                keisan_hoshiki.append(col[6])
        except MySQLdb.OperationalError:
            pass

        arr_header.extend([self.zai_hinname, self.zai_kikaku, self.zai_dansun])
        count = arr_header.count("")
        dansun_empty = False
        dansun_obj = {'dansun_a': dansun_a, 'dansun_b': dansun_b, 'dansun_t': dansun_t, }
        arr_value = {'arr_hinname': hin_name, 'arr_kikaku': kikaku}
        fields_value = {'zai_kikaku': self.zai_kikaku, 'zai_hinname': self.zai_hinname}
        if count >= 2:
            # Handle zaiLengthUnit
            zai_length_unit = "M"
            new_response[zaiLengthUnit] = zai_length_unit
            return new_response
        else:
            length_obj = len(kikaku)
            if not self.zai_hinname:
                response = self.reset_value(len=length_obj, arr_value=arr_value, dansun_obj=dansun_obj,
                                            dansun_empty=dansun_empty, fields_value=fields_value)
                sozaino = self.handle_zai_sozaino()
                new_response[zaiHinname] = response
                new_response[zaiSozaino] = sozaino
            elif not self.zai_kikaku:
                response = self.reset_value(len=length_obj, arr_value=arr_value, dansun_obj=dansun_obj,
                                            dansun_empty=dansun_empty, fields_value=fields_value)
                sozaino = self.handle_zai_sozaino()
                result_hinname = self.get_hinname(self.zai_hinname, hin_name)
                new_response[zaiHinname] = result_hinname
                new_response[zaiKikaku] = response
                new_response[zaiSozaino] = sozaino
            elif not self.zai_dansun:
                dansun_empty = True
                response = self.reset_value(len=length_obj, arr_value=arr_value, dansun_obj=dansun_obj,
                                            dansun_empty=dansun_empty, fields_value=fields_value)
                sozaino = self.handle_zai_sozaino()
                result_hinname = self.get_hinname(self.zai_hinname, hin_name)
                new_response[zaiHinname] = result_hinname
                new_response[zaiDansun] = response
                new_response[zaiSozaino] = sozaino
            else:
                self.reset_value(len=length_obj, arr_value=arr_value, dansun_obj=dansun_obj,
                                 dansun_empty=dansun_empty, fields_value=fields_value)
                result_hinname = self.get_hinname(self.zai_hinname, hin_name)
                new_response[zaiHinname] = result_hinname
                sozaino = self.handle_zai_sozaino()
                new_response[zaiSozaino] = sozaino

            # Handle zaiLengthUnit
            print(zaiLengthUnit)
            zai_length_unit = "M"
            new_response[zaiLengthUnit] = zai_length_unit

        # Handle zai_kikaku
        arr_found_kikaku = get_close_matches(self.zai_kikaku, kikaku)
        if self.zai_kikaku != "":
            if len(arr_found_kikaku) == 0:
                self.zai_kikaku = ""
                new_response[zaiKikaku] = self.zai_kikaku
            else:
                new_response[zaiKikaku] = arr_found_kikaku[0]

        # Handle zai_dansun
        str_dansun = self.zai_dansun
        list_dansun = str_dansun.split("x")
        for i in range(0, len(list_dansun)):
            if list_dansun[i] == "":
                list_dansun[i] = "?"
        new_str_dansun = "x".join(list_dansun)
        new_response[zaiDansun] = new_str_dansun
        return new_response


api.add_resource(HandleRequest, '/ocr_api/alter')

if __name__ == '__main__':
    app.run(debug=True)


# if __name__ == '__main__':
    #     ip_address = 'localhost'
    #     import os
    #
    #     if not DEBUG_MODE:
    #         if not os.path.exists('config'):
    #             print('ip address=')
    #             ip_address = input().strip()
    #             f = open("config", "a")
    #             f.write(ip_address)
    #             f.close()
    #         else:
    #             f = open("config", "r")
    #             ip_address = f.read().strip()
    #
    #     cors = CORS(app, resources={r"/api/*": {"origins": ip_address}})
    #     app.run(debug=DEBUG_MODE, host=ip_address)
