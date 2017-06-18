import datetime
import os
import json
import matplotlib.pyplot as plt


def test1():
    file_date = datetime.date(2014, 6, 7)
    data = loadDay(file_date)
    observations = data['history']['observations']
    for obs in observations:
        temp = obs['tempm']
        vism = obs['vism']
        print(obs['date']['pretty']+" Temp:"+temp+" Visibility:"+vism)


def tempDayData(date):
    data = loadDay(date)
    observations = data['history']['observations']
    temp = []
    for obs in observations:
        temp.append(float(obs['tempm']))
    return temp;
def test2(day,prevdays):
    first = day - +datetime.timedelta(days=prevdays)
    last = day
    i = first
    legendString=[]
    while i <= last:
        plt.plot(tempDayData(i))
        legendString.append(dateString(i))
        i = i +datetime.timedelta(days=1)
    plt.legend(legendString,loc='upper left')
    plt.show()



def dateString(date):
    return f'{date.year}{date.month:02}{date.day:02}'


def loadDay(file_date):
    path = "data/"+dateString(file_date) + '.json'
    if os.path.isfile(path):
        try:
            res_data = json.load(open(path))
            return res_data

        except:
            print("Can not read file:"+path)
            return None


if __name__ == '__main__':
    test1()
   # test2(datetime.date(2017, 6, 16), 7)
