#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Float64, Bool
import time

class JoystickGame():
    def __init__(self):

        self.STATE_WAITING = "waiting"
        self.STATE_SOLVING = "solving"

        self.state = self.STATE_WAITING

        self.input_angle = 0.0
        self.display_time = 0.2 # seconds between display and accepting input

        self.display_timer_start = time.time()

    def setup_topics(self):

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('game_manager', anonymous=True)

        rospy.Subscriber("angle", Float64, self.angle_callback)

        rospy.Subscriber("solution", Bool, self.pomcp_solution_callback)

        self.pub = rospy.Publisher("input_angle", Float64, queue_size=10)

        return


    def angle_callback(self, data):
        """Received an input angle from the angle updater."""

        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

        self.input_angle = data.data

        if self.state == self.STATE_WAITING and time.time()-self.display_timer_start > self.display_time:
            print("Received angle!")
            print("Input: ", self.input_angle, " radians.")

            # Send input to POMCP
            print("Sending input to POMCP")
            self.pub.publish(self.input_angle)

            # update state
            self.state = self.STATE_SOLVING

        return

    def pomcp_solution_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

        if self.state == self.STATE_SOLVING:

            print("Received POMCP solution")

            print("Waiting for display")

            self.state = self.STATE_WAITING
            self.display_timer_start = time.time()

if __name__ == '__main__':

    # Initialize the game
    game = JoystickGame()

    # Setup ros topics
    game.setup_topics()

    # Wait and listen
    rospy.spin()
