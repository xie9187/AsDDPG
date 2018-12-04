import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random

import std_srvs.srv
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

class StageWorld():
	def __init__(self, beam_num):
		 # initiliaze
		rospy.init_node('StageWorld', anonymous=False)

		#------------Params--------------------
		self.image_size = [224, 224]
		self.bridge = CvBridge()

		self.object_state = [0, 0, 0, 0]
		self.object_name = []

		self.self_speed = [0.0, 0.0]
		self.default_states = None
		
		self.start_time = time.time()
		self.max_steps = 10000

		self.scan = None
		self.beam_num = beam_num
		self.laser_cb_num = 0

		self.rot_counter = 0

		self.now_phase = 1
		self.next_phase = 4
		self.step_target = [0., 0.]
		self.step_r_cnt = 0.
		self.stop_counter = 0

		map_img = cv2.imread('./worlds/Obstacles.jpg', 0)
		ret, binary_map = cv2.threshold(map_img,10,1,cv2.THRESH_BINARY)
		binary_map = 1 - binary_map
		# cv2.imshow('img',binary_map*255)
		# cv2.waitKey(0)
		height, width = binary_map.shape
		self.map_pixel = np.array([width, height])
		self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
		self.map = binary_map.astype(np.float32)
		self.raw_map = copy.deepcopy(self.map)
		self.map_origin = self.map_pixel/2 - 1	
		self.R2P = self.map_pixel / self.map_size
		self.robot_size = 0.5
		self.target_size = 0.3

		self.robot_value = .33
		self.target_value = 0.66
		self.path_value = 0.1

		#-----------Publisher and Subscriber-------------
		self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size = 10)

		self.object_state_sub = rospy.Subscriber('base_pose_ground_truth', Odometry, self.GroundTruthCallBack)
		self.laser_sub = rospy.Subscriber('base_scan', LaserScan, self.LaserScanCallBack)
		self.odom_sub = rospy.Subscriber('odom', Odometry, self.OdometryCallBack)
		self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)

		# -----------Service-------------------
		self.ResetStage = rospy.ServiceProxy('reset_positions', std_srvs.srv.Empty)

		# Wait until the first callback
		while self.scan is None:
			pass
		rospy.sleep(1.)
		# What function to call when you ctrl + c    
		rospy.on_shutdown(self.shutdown)


	def GroundTruthCallBack(self, GT_odometry):
		Quaternions = GT_odometry.pose.pose.orientation
		Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
		self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
		v_x = GT_odometry.twist.twist.linear.x
		v_y = GT_odometry.twist.twist.linear.y
		v = np.sqrt(v_x**2 + v_y**2)
		self.speed_GT = [v, GT_odometry.twist.twist.angular.z]


	def ImageCallBack(self, img):
		self.image = img

	def LaserScanCallBack(self, scan):
		self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
						   scan.scan_time, scan.range_min, scan. range_max]
		self.scan = np.array(scan.ranges)
		self.laser_cb_num += 1

	def OdometryCallBack(self, odometry):
		Quaternions = odometry.pose.pose.orientation
		Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
		self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
		self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

	def SimClockCallBack(self, clock):
		self.sim_time = clock.clock.secs + clock.clock.nsecs/1000000000.

	def GetImageObservation(self):
		# ros image to cv2 image
		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.image, "bgr8")
		except Exception as e:
			raise e
		# resize
		dim = (self.image_size[0], self.image_size[1])
		cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
		except Exception as e:
			raise e
		self.resized_ob.publish(resized_img)
		return(cv_resized_img)

	def GetLaserObservation(self):
		scan = copy.deepcopy(self.scan)
		scan[np.isnan(scan)] = 5.6
		scan[np.isinf(scan)] = 5.6
		raw_beam_num = len(scan)
		sparse_beam_num = self.beam_num
		step = float(raw_beam_num) / sparse_beam_num
		sparse_scan_left = []
		index = 0.
		for x in xrange(int(sparse_beam_num/2)):
			sparse_scan_left.append(scan[int(index)])
			index += step
		sparse_scan_right = []
		index = raw_beam_num - 1.
		for x in xrange(int(sparse_beam_num/2)):
			sparse_scan_right.append(scan[int(index)])
			index -= step
		scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
		return scan_sparse / 5.6 - 0.5

	def GetNoisyLaserObservation(self):
		scan = copy.deepcopy(self.scan)
		scan[np.isnan(scan)] = 5.6
		nuniform_noise = np.random.uniform(-0.01, 0.01, scan.shape)
		linear_noise = np.multiply(np.random.normal(0., 0.03, scan.shape), scan) / 5.6
		noise = nuniform_noise + linear_noise
		noise[noise > 0.03] = 0.03
		noise[noise < -0.03] = -0.03
		scan += noise
		scan[scan < 0.] = 0.
		# sample = random.sample(range(0, LAZER_BEAM), LAZER_BEAM/10)
		# scan[sample] = np.random.uniform(0.0, 1.0, LAZER_BEAM/10) * 30.
		return scan

	def GetSelfState(self):
		return self.state;

	def GetSelfStateGT(self):
		return self.state_GT;

	def GetSelfSpeedGT(self):
		return self.speed_GT

	def GetSelfSpeed(self):
		return self.speed

	def GetSimTime(self):
		return self.sim_time


	def ResetWorld(self):
		self.ResetStage()
		self.self_speed = [0.0, 0.0]
		self.step_target = [0., 0.]
		self.step_r_cnt = 0.
		self.start_time = time.time()
		rospy.sleep(0.5)

	def Control(self, action):
		move_cmd = Twist()
		move_cmd.linear.x = action[0]
		move_cmd.linear.y = 0.
		move_cmd.linear.z = 0.
		move_cmd.angular.x = 0.
		move_cmd.angular.y = 0.
		move_cmd.angular.z = action[1]
		self.cmd_vel.publish(move_cmd)

	def shutdown(self):
		# stop turtlebot
		rospy.loginfo("Stop Moving")
		self.cmd_vel.publish(Twist())
		rospy.sleep(1)

	def GetRewardAndTerminate(self, t):
		terminate = False
		reset = False
		laser_scan = self.GetLaserObservation()
		laser_min = np.amin(laser_scan)
		[x, y, theta] =  self.GetSelfStateGT()
		[v, w] = self.GetSelfSpeedGT()
		self.pre_distance = copy.deepcopy(self.distance)
		self.distance = np.sqrt((self.target_point[0] - x)**2 + (self.target_point[1] - y)**2)
		alpha = np.arctan2(self.target_point[1] - y, self.target_point[0] - x) - theta

		# reward = v * np.cos(w) - 0.01
		reward = (self.pre_distance - self.distance) * np.cos(w) - 0.01
		# reward = -0.5 * 0.2
		result = 0
		if v == 0.0 and t > 10 and laser_min < 0.4 / 5.6 - 0.5:
			self.stop_counter += 1
		else:
			self.stop_counter = 0

		if self.distance < self.target_size:
			reward = 5.
			terminate = True
			reset = True
			print 'Reach the Goal'
			result = 3
		else:
			if self.stop_counter == 2 and t <= 200:
				reward = -5.
				terminate = True
				reset = True
				print 'Crash'
				result = 2
			elif t > 200:
				terminate = True
				reset = True
				print 'Time Out'
				result = 1

		return reward, terminate, result

	def GenerateTargetPoint(self):
		x = random.uniform(-(self.map_size[0]/2 - self.target_size), self.map_size[0]/2 - self.target_size)
		y = random.uniform(-(self.map_size[1]/2 - self.target_size), self.map_size[1]/2 - self.target_size)		
		self.target_point = [x, y]
		while not self.TargetPointCheck() and not rospy.is_shutdown():
			x = random.uniform(-(self.map_size[0]/2 - self.target_size), self.map_size[0]/2 - self.target_size)
			y = random.uniform(-(self.map_size[1]/2 - self.target_size), self.map_size[1]/2 - self.target_size)		
			self.target_point = [x, y]
		self.pre_distance = np.sqrt(x**2 + y**2)
		self.distance = copy.deepcopy(self.pre_distance)

	def GetLocalTarget(self):
		[x, y, theta] =  self.GetSelfStateGT()
		[target_x, target_y] = self.target_point
		local_x = (target_x - x) * np.cos(theta) + (target_y - y) * np.sin(theta)
		local_y = -(target_x - x) * np.sin(theta) + (target_y - y) * np.cos(theta)
		return [local_x, local_y]

	def TargetPointCheck(self):
		target_x = self.target_point[0]
		target_y = self.target_point[1]
		pass_flag = True
		x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
		y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
		window_size = int(self.robot_size / 2 * np.amax(self.R2P))
		for x in xrange(np.amax([0, x_pixel - window_size]), np.amin([self.map_pixel[0] - 1, x_pixel + window_size])):
			for y in xrange(np.amax([0, y_pixel - window_size]), np.amin([self.map_pixel[1] - 1, y_pixel + window_size])):
				if self.map[self.map_pixel[1] - y - 1, x] == 1:
					pass_flag = False
					break
			if not pass_flag:
				break
		if abs(target_x) < 2. and abs(target_y) < 2.:
			pass_flag = False
		return pass_flag

	def Global2Local(self, path, pose):
		x = pose[0]
		y = pose[1]
		theta = pose[2]
		local_path = copy.deepcopy(path)
		for t in xrange(0, len(path)):
			local_path[t][0] = (path[t][0] - x) * np.cos(theta) + (path[t][1] - y) * np.sin(theta)
			local_path[t][1] = -(path[t][0] - x) * np.sin(theta) + (path[t][1] - y) * np.cos(theta)
		return local_path

	def ResetMap(self, path):
		self.map = copy.deepcopy(self.raw_map)
		target_point = path[-1]
		self.map = self.DrawPoint(target_point, self.target_size, self.target_value, \
								  self.map, self.map_pixel, self.map_origin, self.R2P)
		return	self.map

	def DrawPoint(self, point, size, value, map_img, map_pixel, map_origin, R2P):
		# x range
		if not isinstance(size, np.ndarray):
			x_range = [np.amax([int((point[0] - size/2) * R2P[0]) + map_origin[0], 0]), \
					   np.amin([int((point[0] + size/2) * R2P[0]) + map_origin[0], \
					   			map_pixel[0] - 1])]

			y_range = [np.amax([int((point[1] - size/2) * R2P[1]) + map_origin[1], 0]), \
					   np.amin([int((point[1] + size/2) * R2P[1]) + map_origin[1], \
					   			map_pixel[1] - 1])]
		else:
			x_range = [np.amax([int((point[0] - size[0]/2) * R2P[0]) + map_origin[0], 0]), \
					   np.amin([int((point[0] + size[0]/2) * R2P[0]) + map_origin[0], \
					   			map_pixel[0] - 1])]

			y_range = [np.amax([int((point[1] - size[1]/2) * R2P[1]) + map_origin[1], 0]), \
					   np.amin([int((point[1] + size[1]/2) * R2P[1]) + map_origin[1], \
					   			map_pixel[1] - 1])]

		for x in xrange(x_range[0], x_range[1] + 1):
			for y in xrange(y_range[0], y_range[1] + 1):
				# if map_img[map_pixel[1] - y - 1, x] < value:
				map_img[map_pixel[1] - y - 1, x] = value
		return map_img	

	def DrawLine(self, point1, point2, value, map_img, map_pixel, map_origin, R2P):
		if point1[0] <= point2[0]:
			init_point = point1
			end_point = point2
		else:
			init_point = point2
			end_point = point1

		# transfer to map point
		map_init_point = [init_point[0] * R2P[0] + map_origin[0], \
						  init_point[1] * R2P[1] + map_origin[1]]
		map_end_point = [end_point[0] * R2P[0] + map_origin[0], \
						 end_point[1] * R2P[1] + map_origin[1]]
		# y = kx + b
		if map_end_point[0] > map_init_point[0]:
			k = (map_end_point[1] - map_init_point[1]) / (map_end_point[0] - map_init_point[0])
			b = map_init_point[1] - k * map_init_point[0]
			if abs(k) < 1.:
				x_range = [np.amax([int(map_init_point[0]), 0]),\
						   np.amin([int(map_end_point[0]), map_pixel[0]])]
				for x in xrange(x_range[0],x_range[1] + 1):
					y = int(x * k + b)
					if y < 0:
						y = 0
					elif y > map_pixel[1]:
						y = map_pixel[1]
					if map_img[map_pixel[1] - y - 1, x] < value:
						map_img[map_pixel[1] - y - 1, x] = value
			else:
				if k > 0:
					y_range = [np.amax([int(map_init_point[1]), 0]),\
							   np.amin([int(map_end_point[1]), map_pixel[1]])]
				else:
					y_range = [np.amax([int(map_end_point[1]), 0]),\
							   np.amin([int(map_init_point[1]), map_pixel[1]])]
				for y in xrange(y_range[0],y_range[1] + 1):
					x = int((y - b)/k)
					if x < 0:
						x = 0
					elif x > map_pixel[0]:
						x = map_pixel[0]
					if map_img[map_pixel[1] - y - 1, x] < value:
						map_img[map_pixel[1] - y - 1, x] = value
		else:
			x_mid = map_end_point[0]
			x_range = [np.amax([int(x_mid - width/2), 0]), \
					   np.amin([int(x_mid + width/2), map_pixel[0]])]
			for x in xrange(x_range[0], x_range[1] + 1):
				y_range = [int(map_init_point[1]), int(map_end_point[1])]
				for y in xrange(y_range[0], y_range[1] + 1):
					map_img[map_pixel[1] - y - 1, x] = value
		return map_img

	def RenderMap(self, path):
		[x, y, theta] =  self.GetSelfStateGT()
		self.ResetMap(path)
		self.map = self.DrawPoint([x, y], self.robot_size, self.robot_value, \
								  self.map, self.map_pixel, self.map_origin, self.R2P)
		return self.map

	def PIDController(self, action_bound):
		X = self.GetSelfState()
		X_t = self.GetLocalTarget()
		# IX_t = 
		# delta_theta = math.atan2(X_t[1] - X[1], X_t[0] - X[0])
		# delta_distance = copy.deepcopy(self.distance)
		# X_r = np.array([delta_distance, delta_theta])
		# X_r = np.array([X_t[0], X_t[1]])
		P = np.array([10, 1])
		Ut = X_t * P

		if Ut[0] < 0.: 
			Ut[0] = 0.
		elif Ut[0]  > action_bound[0]:
			Ut[0] = action_bound[0]

		if Ut[1] < -action_bound[1]: 
			Ut[1] = -action_bound[1]
		elif Ut[1]  > action_bound[1]:
			Ut[1] = action_bound[1]	

		return [Ut]
# env = StageWorld(10)
# print env.GetLaserObservation()
