import datetime
import numpy as np
import data_view as d
import random
import data_test as t
import pathlib

PARAMS_COUNT=8
FIRST_DATA = datetime.date(2014, 6, 7)
TODAY = datetime.date.today()

def checkData(date):
    file = t.filename(date)
    if not pathlib.Path("data/" + file+".json").is_file():
        raise Exception('no data')



def getObservationData(obs):
    data = np.zeros([PARAMS_COUNT])
    data[0] = float(obs['tempm'])
    data[1] = (float(obs['dewptm']))
    data[2] = (float(obs['vism']))
    data[3] = (float(obs['pressurem']))
    try:
        data[4] = (float(obs['hum']))
    except:
        #Sometimes input data have humidity as N/A then set it to zero
        data[4]= 0.0
    data[5] = (float(obs['fog']))
    data[6] = (float(obs['snow']))
    data[7] = (float(obs['thunder']))
    return data;

def parseData(date):
    data = d.loadDay(date)
    try:
        observations = data['history']['observations']
    except:
        print("Problem: data['history']['observations']:", d.dateString(date))
    for obs in observations:
        hour = float(obs['date']['hour'])
        if hour == 1:
            input1 = getObservationData(obs)
        if hour == 12:
            input12 = getObservationData(obs)
        if hour == 19:
            input19 = getObservationData(obs)
    return (input1,input12,input19)

def getOneDay(date):
    """
     It returns single list of data which
     can be provided as input to neural network
     This data contains weather  about one day to
     predict temperature to following day.
     This data contains
     1. Temperature(tempm)
     2. Dew point(dewptm)
     3. Visibility(vism)
     4. Pressure(pressurem)
     5. Humidity(hum)
     6. Fog(fog)
     7. Snow(snow)
     8. Thunder(thunder)

     For three  specific hours: 1,12,19

    :param date: specify day to get data for example datetime.date(2014, 6, 7)
    :return: two vector first is input weather data and output is only
    temperature for three hours 1,12,19
    """

    input1,input12,input19 = parseData(date)
    input_data = np.array([input1, input12, input19]).flatten()
    input1, input12, input19 = parseData(date+datetime.timedelta(days=1))
    output = [input1[0], input12[0], input19[0]]
    return [input_data, output]

def getBatchDays(date,previousdays):
    result_input = np.zeros([previousdays, 3*PARAMS_COUNT])
    result_output = np.zeros([previousdays, 3])
    iter_day = date-datetime.timedelta(days=previousdays)
    row =0
    while iter_day < date:
        result_input[row], result_output[row] = getOneDay(iter_day)
        row += 1
        iter_day = iter_day + datetime.timedelta(days=1)
    return (result_input, result_output)

def getRandomTrainBatch(batch_size):
    rand = random.SystemRandom()
    start = FIRST_DATA + datetime.timedelta(days=batch_size)
    end = TODAY - datetime.timedelta(days=1)
    while True:
        try:
            day = int(random.SystemRandom.uniform(rand, 1, 31))
            month = int(random.SystemRandom.uniform(rand, 1, 12))
            year = int(random.SystemRandom.uniform(rand,start.year, end.year+1))
            date = datetime.date(year, month, day)
            if(date < start):
                continue
            checkData(date)

            break
        except:
            continue
    return getBatchDays(date,batch_size)
if __name__ == '__main__':
    #input_data, output = getOneDay(datetime.date(2014, 6, 7))
    #input_data, output = getBatchDays(datetime.date(2015, 6, 7),20)
    input_data, output = getRandomTrainBatch(2)
    print("Input data:")
    print(input_data)
    print("Output data:")
    print(output)
