import pathlib
import calendar
import datetime

calendar.setfirstweekday(calendar.MONDAY)

DATA_DIR="data/"


FIRST = datetime.date(2014, 6, 7)
TODAY = datetime.date.today()

missingFiles = []
DATE = FIRST
while DATE <TODAY :
    file = f'{DATE.year}{DATE.month:02}{DATE.day:02}.json'
    DATE = DATE+datetime.timedelta(days=1)
    if not pathlib.Path(DATA_DIR + file).is_file():
        print(file, '-missing')
        missingFiles.append(file)
