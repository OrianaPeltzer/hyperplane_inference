#!/usr/bin/env python
"""
Listens to joystick instructions, updates the angle entered from the user,
and publishes the angle to rostopic /angle.
"""

import rospy
import pygame
import os
import pprint

from std_msgs.msg import Int32, Float64

from pyPS4Controller.controller import Controller
from pyPS4Controller.event_mapping.Mapping3Bh2b import Mapping3Bh2b
import numpy as np

class MyController(Controller):

    def __init__(self, publisher, **kwargs):

        self.controller = None
        self.axis_data = None
        self.button_data = None
        self.hat_data = None

        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        #Controller.__init__(self, **kwargs)

        self.R3_joystick_coordinates = np.array([0.0,0.0])
        self.R3_angle = 0.0
        self.pub = publisher

    def listen(self):
        """Listen for events to happen"""
        
        if not self.axis_data:
            self.axis_data = {}

        if not self.button_data:
            self.button_data = {}
            for i in range(self.controller.get_numbuttons()):
                self.button_data[i] = False

        if not self.hat_data:
            self.hat_data = {}
            for i in range(self.controller.get_numhats()):
                self.hat_data[i] = (0, 0)

        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.axis_data[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                self.button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                self.button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                self.hat_data[event.hat] = event.value

            # Insert your code on what you would like to happen for each event here!
            # In the current setup, I have the state simply printing out to the screen.
                
            #os.system('clear')
            print("\n New Controller Instance \n")
            # pprint.pprint(self.button_data)
            pprint.pprint(self.axis_data)
            # pprint.pprint(self.hat_data)

            if self.axis_data:
                self.R3_joystick_coordinates[0] = self.axis_data.get(2, 0.0) # x (-1 --> 1)
                self.R3_joystick_coordinates[1] = -self.axis_data.get(3,0.0) # y is inverted: -1 is up

                self.print_joystick_coordinates()
                self.detect_joystick()

    def on_x_press(self):
       print("Hello world")

    def on_x_release(self):
       print("Goodbye world")

    def on_R3_up(self, value):
        # print("on_R3_up: {}".format(float(value)/32500.0))
        self.R3_joystick_coordinates[1] = -float(value)/32500.0
        self.detect_joystick()
        # self.print_joystick_coordinates()

    def on_R3_down(self, value):
        # print("on_R3_down: {}".format(float(value)/32500.0))
        self.R3_joystick_coordinates[1] = -float(value)/32500.0
        self.detect_joystick()
        # self.print_joystick_coordinates()

    def on_R3_left(self, value):
        # print("on_R3_left: {}".format(float(value)/32500.0))
        self.R3_joystick_coordinates[0] = float(value)/32500.0
        self.detect_joystick()
        # self.print_joystick_coordinates()

    def on_R3_right(self, value):
        # print("on_R3_right: {}".format(float(value)/32500.0))
        self.R3_joystick_coordinates[0] = float(value)/32500.0
        self.detect_joystick()
        # self.print_joystick_coordinates()

    def on_R3_y_at_rest(self):
        """R3 joystick is at rest after the joystick was moved and let go off"""
        print("on_R3_y_at_rest")

    def on_R3_x_at_rest(self):
        """R3 joystick is at rest after the joystick was moved and let go off"""
        print("on_R3_x_at_rest")

    def print_joystick_coordinates(self):
        print("Joystick coordinates: ", self.R3_joystick_coordinates)

    def detect_joystick(self):
        """Detects whether there is a joystick command, and if so, updates and
        displays the angle."""

        if np.linalg.norm(self.R3_joystick_coordinates) > 0.9:
            print("Joystick Event Detected!")
            self.R3_angle = np.arctan2(self.R3_joystick_coordinates[1], self.R3_joystick_coordinates[0])
            print("Angle: ", self.R3_angle)
            self.pub.publish(self.R3_angle)
        return

    def _Controller__handle_event(self, button_id, button_type, value, overflow, debug):

        event = self.event_definition(button_id=button_id,
                                      button_type=button_type,
                                      value=value,
                                      connecting_using_ds4drv=self.connecting_using_ds4drv,
                                      overflow=overflow,
                                      debug=debug)

        if event.R3_event():
            self.event_history.append("right_joystick")
            if event.R3_y_at_rest():
                self.on_R3_y_at_rest()
            elif event.R3_x_at_rest():
                self.on_R3_x_at_rest()
            elif event.R3_right():
                self.on_R3_right(event.value)
            elif event.R3_left():
                self.on_R3_left(event.value)
            elif event.R3_up():
                self.on_R3_up(event.value)
            elif event.R3_down():
                self.on_R3_down(event.value)
        elif event.L3_event():
            self.event_history.append("left_joystick")
            if event.L3_y_at_rest():
                self.on_L3_y_at_rest()
            elif event.L3_x_at_rest():
                self.on_L3_x_at_rest()
            elif event.L3_up():
                self.on_L3_up(event.value)
            elif event.L3_down():
                self.on_L3_down(event.value)
            elif event.L3_left():
                self.on_L3_left(event.value)
            elif event.L3_right():
                self.on_L3_right(event.value)
        elif event.circle_pressed():
            self.event_history.append("circle")
            self.on_circle_press()
        elif event.circle_released():
            self.on_circle_release()
        elif event.x_pressed():
            self.event_history.append("x")
            self.on_x_press()
        elif event.x_released():
            self.on_x_release()
        elif event.triangle_pressed():
            self.event_history.append("triangle")
            self.on_triangle_press()
        elif event.triangle_released():
            self.on_triangle_release()
        elif event.square_pressed():
            self.event_history.append("square")
            self.on_square_press()
        elif event.square_released():
            self.on_square_release()
        elif event.L1_pressed():
            self.event_history.append("L1")
            self.on_L1_press()
        elif event.L1_released():
            self.on_L1_release()
        elif event.L2_pressed():
            self.event_history.append("L2")
            self.on_L2_press(event.value)
        elif event.L2_released():
            self.on_L2_release()
        elif event.R1_pressed():
            self.event_history.append("R1")
            self.on_R1_press()
        elif event.R1_released():
            self.on_R1_release()
        elif event.R2_pressed():
            self.event_history.append("R2")
            self.on_R2_press(event.value)
        elif event.R2_released():
            self.on_R2_release()
        elif event.options_pressed():
            self.event_history.append("options")
            self.on_options_press()
        elif event.options_released():
            self.on_options_release()
        elif event.left_right_arrow_released():
            self.on_left_right_arrow_release()
        elif event.up_down_arrow_released():
            self.on_up_down_arrow_release()
        elif event.left_arrow_pressed():
            self.event_history.append("left")
            self.on_left_arrow_press()
        elif event.right_arrow_pressed():
            self.event_history.append("right")
            self.on_right_arrow_press()
        elif event.up_arrow_pressed():
            self.event_history.append("up")
            self.on_up_arrow_press()
        elif event.down_arrow_pressed():
            self.event_history.append("down")
            self.on_down_arrow_press()
        elif event.playstation_button_pressed():
            self.event_history.append("ps")
            self.on_playstation_button_press()
        elif event.playstation_button_released():
            self.on_playstation_button_release()
        elif event.share_pressed():
            self.event_history.append("share")
            self.on_share_press()
        elif event.share_released():
            self.on_share_release()
        elif event.R3_pressed():
            self.event_history.append("R3")
            self.on_R3_press()
        elif event.R3_released():
            self.on_R3_release()
        elif event.L3_pressed():
            self.event_history.append("L3")
            self.on_L3_press()
        elif event.L3_released():
            self.on_L3_release()


if __name__=='__main__':
    rospy.init_node('angle_updater')
    pub=rospy.Publisher('angle', Float64, queue_size=10)
    rate= rospy.Rate(5)
    controller = MyController(publisher=pub, interface="/dev/input/js0", connecting_using_ds4drv=False, event_definition=Mapping3Bh2b)

    while not rospy.is_shutdown():

        # you can start listening before controller is paired, as long as you pair it within the timeout window
        controller.listen()
        rate.sleep()

