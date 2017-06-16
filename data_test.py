import pathlib
import calendar
import datetime
import os
import sys
import requests
import json

calendar.setfirstweekday(calendar.MONDAY)

DATA_DIR = "data/"
FIRST = datetime.date(2014, 6, 7)
TODAY = datetime.date.today()

missingFiles = []


def filename(date):
    return f'{date.year}{date.month:02}{date.day:02}'


def checkData():
    date = FIRST
    while date < TODAY:
        date = date+datetime.timedelta(days=1)
        file = filename(date)
        if not pathlib.Path(DATA_DIR + file+".json").is_file():
            print(file, '-missing')
            missingFiles.append(date)


def getMissingData():
    # Read API key from environment
    if 'WUNDERGROUND_APIKEY' in os.environ:
        API_KEY = os.environ.get('WUNDERGROUND_APIKEY')
    else:
        print("APIKEY not found")
        sys.exit(1)
    for x in missingFiles:
        date = filename(x)
        query_link = f'http://api.wunderground.com/api/{API_KEY}/history_{date}/q/zmw:00000.1.12566.json'
        request = requests.get(query_link)
        data = request.json()
        try:
            file = DATA_DIR+date+".json"
            out = open(file, 'w')
            json.dump(data, out)
            out.close()
            print("File created:"+file)
        except:
            print("Unable to write file:"+date);

if __name__ == '__main__':
    checkData()
    #os.environ.setdefault("WUNDERGROUND_APIKEY", "1")
    getMissingData()
