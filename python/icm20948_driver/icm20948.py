#!/usr/bin/python3
import smbus
import math
import numpy as np
from threading import Thread

WHO_AM_I = 0x00

class ICM20948():
    """
    ICM20948 I2C driver [1]_ [1]_

    :param str nav_frame: navigation frame
    :param int axis: axis data
    :param float hz: IMU frequency
    :param bool calibration: calibrate gyroscope and accelerometer

    .. Reference
    .. [1] 'ICM-20948 Datasheets <https://invensense.tdk.com/wp-content/uploads/2024/03/DS-000189-ICM-20948-v1.6.pdf>'
    """

    def __init__(self, nav_frame, axis, hz, calibration):
        # I2C connection parameter
        self.icm20948_address = 0x68
        self.bus = smbus.SMBus(1)
        address_list = self.check_i2c_address()
        print(address_list)
        if len(address_list) == 0:
            raise RuntimeError("No i2c address found")
        else: 
            print("Default ICM20948 address is 0x68")
    
        # driver parameter
        self.nav_frame = nav_frame
        self.axis = axis
        self.hz = hz
        self.dt = 1/self.hz
        self.queue_size = 20
        self.window_size = 5
        self.gyro_queue = np.empty([1,3])
        self.calibration = calibration

        # Check parameter
        if (self.axis != 6) and (self.axis != 9):
            raise ValueError("Axis must be 6 or 9")
        if self.nav_frame=="NED":
            self.body_frame = "FRD"
            self.rotation_seq = "zyx"
        elif self.nav_frame=="ENU":
            self.body_frame = "RFU"
            self.rotation_seq = "zxy"
        else:
            raise ValueError("Navigation frame should be either ENU or NED")

        # Config ICM20948
        self.who_am_i()


    def check_i2c_address(self):
        """
        Search all I2C addresses and record accessible addresses

        :returns: 
            - address_list (list) - stores a list of all accessible I2C addresses
        """
        address_list = []
        for device in range(128):
            try:
                self.bus.read_byte(device)
                address_list.append(hex(device))
            except:
                pass
        return address_list
    
    def who_am_i(self):
        """
        Check ICM20948 WHOAMI register value
        """
        value = hex(self.read_8bit_register(WHO_AM_I))
        print("The register value is {}".format(value))
        if value == "0xEA":
            print("It is ICM20948 default value")
        else:
            print("It is not ICM20948 default value")
            raise RuntimeError("ICM20948 not found")
        
    def read_8bit_register(self, single_register):
        """
        Access the registers and return its raw value

        :param int single_register: single registers address

        :returns: 
            - signed_value (int) - sensor value in int16 format
        """
        value = self.bus.read_byte_data(self.icm20948_address, single_register)
        return value
        # while True:
        #     try:
        #         value = self.bus.read_byte_data(self.address, single_register)
        #         return value
        #     except:
        #         print("\nICM20948 register error occur")
        #         continue
