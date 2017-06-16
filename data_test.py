import pathlib
import calendar
import datetime
import os
import sys

calendar.setfirstweekday(calendar.MONDAY)

DATA_DIR = "data/"
FIRST = datetime.date(2014, 6, 7)
TODAY = datetime.date.today()

missingFiles = []


def checkData():
    date = FIRST
    while date < TODAY:
        file = f'{date.year}{date.month:02}{date.day:02}.json'
        date = date+datetime.timedelta(days=1)
        if not pathlib.Path(DATA_DIR + file).is_file():
            print(file, '-missing')
            missingFiles.append(file)


def getMissingData():
    # Read API key from environment
    if 'WUNDERGROUND_APIKEY' in os.environ:
        API_KEY = os.environ.get('WUNDERGROUND_APIKEY')
    else:
        print("APIKEY not found")
        sys.exit(1)

if __name__ == '__main__':
    checkData()
    #os.environ.setdefault("WUNDERGROUND_APIKEY", "1")
    getMissingData()
