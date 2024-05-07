
import time
import RPi.GPIO as GPIO
PUL_L = 17  # Stepper Drive Pulses
DIR_L = 27  # Controller Direction Bit (High for Controller default / LOW to Force a Direction Change).
ENA_L= 22  # Controller Enable Bit (High to Enable / LOW to Disable).
PUL_R = 16  # Stepper Drive Pulses
DIR_R = 20  # Controller Direction Bit (High for Controller default / LOW to Force a Direction Change).
ENA_R = 21  # Controller Enable Bit (High to Enable / LOW to Disable).

GPIO.setmode(GPIO.BCM)

GPIO.setup(PUL_L, GPIO.OUT)
GPIO.setup(DIR_L, GPIO.OUT)
GPIO.setup(ENA_L, GPIO.OUT)
GPIO.setup(PUL_R, GPIO.OUT)
GPIO.setup(DIR_R, GPIO.OUT)
GPIO.setup(ENA_R, GPIO.OUT)


delay = 0.001

durationFwd = 100

F = 1
B = 0

def forward():
    GPIO.output(ENA_L, GPIO.HIGH)
    GPIO.output(ENA_R, GPIO.HIGH)
    #GPIO.output(ENAI, GPIO.HIGH)
    print('ENA set to HIGH - Left Controller Enabled')
        
    sleep(.5) # pause due to a possible change direction
    GPIO.output(DIR_L, GPIO.LOW)
    GPIO.output(DIR_R, GPIO.LOW)
    #GPIO.output(DIRI, GPIO.LOW)
    print('DIR set to LOW - Moving Forward at ' + str(delay))
    print('Controller PUL being driven.')
    for x in range(durationFwd):
        GPIO.output(PUL_L, GPIO.HIGH)
        GPIO.output(PUL_L, GPIO.HIGH)
        sleep(delay)
        GPIO.output(PUL_L, GPIO.LOW)
        GPIO.output(PUL_L, GPIO.LOW)
        sleep(delay)
    GPIO.output(ENA_L, GPIO.LOW)
    #GPIO.output(ENAI, GPIO.LOW)
    print('ENA set to LOW - Left Controller Disabled')
    sleep(.5) # pause for possible change direction
    return

def leftTurn(): pass
    
def rightTurn():pass
    
def extremeleftTurn():pass

def extremerightTurn():pass

