#from serial import Serial
import serial
import io
from PIL import Image
import numpy as np
import base64
import time

def float_2_fixed_len_string(number, len, presicion):
    instruction = '{:+' + str(len) + '.' + str(presicion) + 'f}'
    string = instruction.format(number)
    return string


ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM4'
ser.timeout=1

try:
    ser.open()
except serial.SerialException:
    print("Unable to open Port")


camID = '1'
alfa1 = 2.4556264582
beta1= 9.123456789
alfa2 = -4.4556264582
beta2= 5.123456789

while (True):
    ser.write('xxPleaseSYNCwithMExx'.encode())
    ser.write(float_2_fixed_len_string(alfa1, 10, 6).encode())
    ser.write('@'.encode())
    ser.write(float_2_fixed_len_string(beta1, 10, 6).encode())
    ser.write('@'.encode())
    ser.write(float_2_fixed_len_string(alfa2, 10, 6).encode())
    ser.write('@'.encode())
    ser.write(float_2_fixed_len_string(beta2, 10, 6).encode())
    ser.write('@'.encode())
    ser.write('xxPleaseSYNCwithMExx'.encode())
    time.sleep(.4)
    print(str('xxPleaseSYNCwithMExx')+camID+float_2_fixed_len_string(alfa1,10,6)+'@'+float_2_fixed_len_string(beta1, 10, 6)+'@'+float_2_fixed_len_string(alfa2,10,6)+'@'+float_2_fixed_len_string(beta2, 10, 6)+'@')

if ser.is_open:
    ser.close()

