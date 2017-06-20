# NN-weather

Neuar network model to weather predictions build with tensor flow.
It is based on LSTM cell.
To execute model just enter this line:
python nn_weather.py --save_path=./model.ckpt

To load earlier trained model:
 python nn_weather.py --state_path=./model.ckpt

It also creates files form data buffer in /tmp/proj
