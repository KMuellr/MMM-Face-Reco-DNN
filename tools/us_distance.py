"""Provides the 'get_distance' function which uses the ultrasonic sensor to 
determine the distance from the mirror to the nearest object."""
import time
import RPi.GPIO as GPIO

# setup pins
GPIO.setmode(GPIO.BCM)
TRIG = 4
ECHO = 17
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.output(TRIG, False)
time.sleep(1) # give everything some time to settle

def get_distance():
    """Read the distance from the ultrasonic sensor.

    Returns
    -------
    float
        The obtained distance in cm.
    """
    n = 10
    recv_list_0 = list(n*(0,)) # last n received inputs
    recv_list_1 = list(n*(1,))
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    while sum(recv_list_0)/n < 1:
        recv_list_0.append(GPIO.input(ECHO))
        recv_list_0.pop(0)
    pulse_start = time.time()
    while sum(recv_list_1) > 0:
        recv_list_1.append(GPIO.input(ECHO))
        recv_list_1.pop(0)
    pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)

def cleanup():
    """Calls GPIO.cleanup()"""
    GPIO.cleanup()

if __name__ == "__main__":
    print("[press ctrl+c to end the script]")
    try: # Main program loop
        while True:
            distance = get_distance()
            print("Distance is {} cm".format(distance))
            time.sleep(3)
        # Scavenging work after the end of the program
    except KeyboardInterrupt:
        print("Script end!")
    finally:
        cleanup()