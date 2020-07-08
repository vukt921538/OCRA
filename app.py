from flask import Flask
from flask_restful import Resource, Api
import MySQLdb
import json
from difflib import get_close_matches

app = Flask(__name__)
api = Api(app)

db = MySQLdb.connect("localhost", "root", "123456", "morisada", charset="utf8")

zaiLengthUnit = "zaiLengthUnit"
zaiHinname = "zaiHinname"
zaiKikaku = "zaiKikaku"
zaiDansun = "zaiDansun"
zaiLength = "zaiLength"
zaiInzu = "zaiInzu"
zaiSozaiSeizobi = "zaiSozaiSeizobi"
zaiSozaino = "zaiSozaino"
zaiCharge = "zaiCharge"

class HandleRequest(Resource):

    def __init__(self):
        data = self.read_json_file()
        self.zai_length_unit = data[zaiLengthUnit]
        self.zai_hinname = data[zaiHinname]
        self.zai_kikaku = data[zaiKikaku]
        self.zai_dansun = data[zaiDansun]
        self.zai_length = data[zaiLength]
        self.zai_inzu = data[zaiInzu]
        self.zai_sozaiseizobi = data[zaiSozaiSeizobi]
        self.zai_sozaino = data[zaiSozaino]
        self.zai_charge = data[zaiCharge]

    @staticmethod
    def read_json_file():
        with open('img_content.json', encoding='utf-8') as json_file:
            result = json.load(json_file)
            return result

    def handle_data_receiver(self):
        data = self.read_json_file()
        return data

    def get(self):
        new_response = self.handle_data_receiver()
        print("JSON: ", new_response)
        # create 7 empty list
        kihon_tanju_cd, kikaku, hin_name, dansun_t, dansun_a, dansun_b, keisan_hoshiki = ([] for i in range(7))
        # get response
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

        db.close()

        # Handle logic
        zai_hinname = self.zai_hinname
        zai_kikaku = self.zai_kikaku
        zai_dansun = self.zai_dansun
        if zai_hinname or zai_kikaku or zai_dansun:
            length_obj = len(kikaku)
            if not zai_hinname:
                same_value = []
                for i in range(0, length_obj):
                    new_dansun = str(int(dansun_a[i])) + "x" + str(int(dansun_b[i])) + "x" + str(int(dansun_t[i]))
                    if (zai_kikaku == kikaku[i]) and (zai_dansun == new_dansun):
                        # GET hin_name
                        same_value.append(hin_name[i])
                new_response[zaiHinname] = same_value[0]
            elif not zai_kikaku:
                same_value = []
                for i in range(0, length_obj):
                    new_dansun = str(int(dansun_a[i])) + "x" + str(int(dansun_b[i])) + "x" + str(int(dansun_t[i]))
                    if (zai_hinname == hin_name[i]) and (zai_dansun == new_dansun):
                        # GET kikaku
                        same_value.append(kikaku[i])
                new_response[zaiKikaku] = same_value[0]
            elif not zai_dansun:
                same_value = []
                for i in range(0, length_obj):
                    if (zai_hinname == hin_name[i]) and (zai_kikaku == kikaku[i]):
                        # GET dansun
                        new_dansun = str(int(dansun_a[i])) + "x" + str(int(dansun_b[i])) + "x" + str(int(dansun_t[i]))
                        same_value.append(new_dansun)
                new_response[zaiDansun] = same_value[0]

            # if zai_hinname:  # String not Empty
            #     # Get similar string in datatbase
            #     most_similar_string_zai_hinname = get_close_matches(zai_hinname, hin_name)
            #     print(most_similar_string_zai_hinname)
            # else:
            #     print("EMPTY")
        else:
            print("FAF")
            pass
        return new_response


api.add_resource(HandleRequest, '/ocr_api')

if __name__ == '__main__':
    app.run(debug=True)
