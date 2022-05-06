#!/usr/bin/env python
"""
	Sample boiler-plate code for phase 1b
	Cyber Physical System Virtual Organization Challenge 2021 : SoilScope Lunar Lander ExoCam -- Earth Analog
	Team Name :
	Members :
"""

import rospy
import numpy as np
import math
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Vector3Stamped, Quaternion
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget, ActuatorControl, RCOut
from sensor_msgs.msg import Imu, Image
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from gazebo_ros_link_attacher.srv import Attach, AttachRequest

from mavros_msgs.srv import SetMode, CommandBool
from std_msgs.msg import String, Header, Bool, Float64
from numpy import sqrt, pi, sin, cos, arctan2, array, linalg, tan, dot

from std_srvs.srv import Empty

g = 9.80665

class OffboardControl:
	""" Controller for PX4-UAV offboard mode """

	def __init__(self):
		rospy.init_node('OffboardControl', anonymous=True)

		# define your class variables here

		self.curr_pose = PoseStamped()                      # current pose of the drone
		self.des_pose = PoseStamped()                           # desired pose of the drone in position control mode
		self.is_ready_to_fly = False
		self.mode = "ASCEND"
		self.arm = False
		self.att = AttitudeTarget()
		self.attach = False
		self.orientation = [0]*3
		self.orientation_mapped = [0] * 3
		self.attachFlag = True        
		self.parachuteFL = False
		self.rate = rospy.Rate(120)


		self.initializeVisionVariables()

		for i in reversed(range(1,4)):
			print("Launching node in {}...".format(i))
			rospy.sleep(1)

		# define ros services, subscribers and publishers here
		# arm or disarm the UAV
		self.armService = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
		# attach any two objects in Gazebo
		self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
		# detach any two attached objects
		self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
		# pause the Gazebo simulation if needed (could be used to debug the movement of the UAV)
		self.pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		# example call:
#		self.pause_physics_client.call()
		# could be used to reset the probe if needed
		self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)

		# command your attitude target to the UAV
		self.att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
		# command a position setpoint to the UAV
		self.pos_setpoint_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
		# publish the debug image for reference if needed
		self.debugImgPub = rospy.Publisher('/debug_cam',Image,queue_size=10)

		# get the current state of the UAV
		self.state_sub = rospy.Subscriber('mavros/state', State, callback=self.state_callback)
		
		self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.pose_callback)
		# get the visual from the onboard camera
		self.img_sub = rospy.Subscriber('/uav_camera_down/image_raw',Image,self.img_cb)


		# call the state machine
		self.controller()

	def initializeVisionVariables(self):
		self.bridge = CvBridge()
		self.debug = True
		self.imgSize = array([640,640,3])

	def pose_callback(self, msg):
		self.curr_pose = msg
		# gets the euler angles (roll, pitch, yaw) from the quaternion values
		# Note: The angles you get doesn't map the [-pi,pi] range properly and might require some conversion
		self.orientation = euler_from_quaternion((msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w))
		self.orientation_mapped[0] = self.orientation[0]*180/math.pi
		self.orientation_mapped[1] = self.orientation[1]*180/math.pi
		self.orientation_mapped[2] = self.orientation[2]*180/math.pi
		# print(self.orientation_mapped[1])


	def img_cb(self,msg):
		try:
			if self.curr_pose.pose.position.z > 0:
				# access the visual from 'frame' to get the rover coordinates
				frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
				# add your image processing code here
				# hsv filtering
				hMin, sMin, vMin = 80,0,0
				hMax, sMax, vMax = 150,100,100
				lower = np.array([hMin, sMin, vMin])
				upper = np.array([hMax, sMax, vMax])
				im_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
				green_mask = cv.inRange(im_hsv, lower, upper)
				green_frame = cv.bitwise_and(frame, frame, mask=green_mask)
				cv.imshow("color masked", green_frame)
				cv.waitKey(0)
				# finding contours
				_, contours, _ = cv.findContours(green_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
				# Find the index of the largest contour
				areas = [cv.contourArea(c) for c in contours]
				max_index = np.argmax(areas)
				cnt=contours[max_index]
				cv.drawContours(frame, [cnt], -1, (0,255,0), 2)
				cv.imshow("with contours", green_frame)
				cv.waitKey(0)
				#draw bounding box
				x,y,w,h = cv.boundingRect(cnt)
				cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				
				if self.debug:
					# could be used to debug your detection logic
					data = self.bridge.cv2_to_imgmsg(frame,"bgr8")
					data.header.stamp = msg.header.stamp
					self.debugImgPub.publish(data)
		except CvBridgeError as e:
			print(e)
			pass



	def state_callback(self, msg):
		if msg.mode != 'OFFBOARD' or self.arm != True:
			# take_off
			self.set_offboard_mode()
			self.set_arm()

	def set_offboard_mode(self):
		rospy.wait_for_service('/mavros/set_mode')
		try:
			flightModeService = rospy.ServiceProxy('/mavros/set_mode', SetMode)
			isModeChanged = flightModeService(custom_mode='OFFBOARD')
		except rospy.ServiceException as e:
			print("service set_mode call failed: %s. OFFBOARD Mode could not be set. Check that GPS is enabled" % e)

	def set_arm(self):
		rospy.wait_for_service('/mavros/cmd/arming')
		try:
			self.armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
			self.armService(True)
			self.arm = True
		except rospy.ServiceException as e:
			print("Service arm call failed: %s" % e)



	def attach_models(self, model1, link1, model2, link2):
		req = AttachRequest()
		req.model_name_1 = model1
		req.link_name_1 = link1
		req.model_name_2 = model2
		req.link_name_2 = link2
		self.attach_srv.call(req)

	def detach_models(self, model1, link1, model2, link2):
		req = AttachRequest()
		req.model_name_1 = model1
		req.link_name_1 = link1
		req.model_name_2 = model2
		req.link_name_2 = link2
		self.detach_srv.call(req)


	def retro(self):
		while self.mode == 'RETRO' and not rospy.is_shutdown():
			# using the following line of code to detach the probe
			self.detach_models('if750a','base_link','sample_probe','base_link')
			# self.parachuteFL = True
			self.mode = "LAND"
			

	def land(self):
		while self.mode == "LAND" and not rospy.is_shutdown():
			try:
				flightModeService = rospy.ServiceProxy('/mavros/set_mode', SetMode)
				isModeChanged = flightModeService(custom_mode='AUTO.LAND')
			except rospy.ServiceException as e:
				print("service set_mode call failed: %s. OFFBOARD Mode could not be set. Check that GPS is enabled" % e)

			# use the following code to pull up the parachute on the probe so that it lands on the spot!
			# if self.parachuteFL is True:
			# 	self.parachuteFL=False
			# 	rospy.sleep(1.2)
			# 	self.parachute_pub.publish(1)

			try:  # prevent garbage in console output when thread is killed
				self.rate.sleep()
			except rospy.ROSInterruptException:
				pass

	def ascend(self):
		self.des_pose.pose.position.z = 10
		while self.mode=="ASCEND" and not rospy.is_shutdown():
			self.pos_setpoint_pub.publish(self.des_pose)
			if self.curr_pose.pose.position.z>0.5 and self.attachFlag:
				# Use the following line of code to attach the probe to the drone...
				self.attach_models('if750a','base_link','sample_probe','base_link')
				self.attachFlag = False
				print('------------Attached the probe------------')
			# if the drone is ready for the throw, change mode to "BELLY-FLOP"
			# if self.curr_pose.pose.position.z > 10:
			# 	self.mode = "BELLY-FLOP"
			self.rate.sleep()

	def belly_flop(self):

		# add your flip code here 

		self.set_offboard_mode()
		self.att.body_rate = Vector3()
		self.att.header = Header()
		self.att.header.frame_id = "base_footprint"
		self.attach = True
		count=0

		while self.mode == "BELLY-FLOP" and not rospy.is_shutdown():
			self.att.header.stamp = rospy.Time.now()
			# use AttitudeTarget.thrust to lift your quadcopter
			self.att.thrust = 0.7
			# use AttitudeTarget.body_rate.y to provide the angular velocity to your quadcopter
			self.att.body_rate.y = 0.0
			# type_mask = 128 is used for controlling the rate exclusively, you may explore other values too
			self.att.type_mask = 128

			# if (you think the drone is ready to detach the probe):
			if self.orientation_mapped[1] >= 35 and self.orientation_mapped[1] <= 55: # 0.785(45) 2.267(130) and 2.442(140) 2.529(145)
                # self.att_setpoint_pub.publish(self.att)
                # self.rate.sleep()
                # self.mode="RETRO"
				print(count)
				if count >= 5:
					print("................................................   " +  str(self.orientation_mapped[1]))
					self.att_setpoint_pub.publish(self.att)
					self.rate.sleep()
					self.mode="RETRO"
				count += 1

			self.att_setpoint_pub.publish(self.att)

			try:  # prevent garbage in console output when thread is killed
				self.rate.sleep()
			except rospy.ROSInterruptException:
				pass



	def controller(self):
		""" A state machine developed to have UAV states """
		while not rospy.is_shutdown():
			# control your UAV states and functionalities here...
			if self.mode =="ASCEND":
				print("Ascending!")
				self.ascend()
			if self.mode =="BELLY-FLOP":
				print("belly flop!")
				self.belly_flop()
			if self.mode == "RETRO":
				self.retro()
			if self.mode == "LAND":
				self.land()


if __name__ == "__main__":
	
	OffboardControl()
