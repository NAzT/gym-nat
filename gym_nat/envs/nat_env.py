import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pyglet
from makerasia import pendulum

import math


class DrawText:
	def __init__(self, label: pyglet.text.Label):
		self.label = label

	def render(self):
		self.label.draw()


class NatEnv2(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 50
	}

	def __init__(self, g=10.0):
		self.deg = 0
		self.max_speed = 8
		self.max_torque = 2.
		self.dt = .05
		self.g = g
		self.m = 1.
		self.l = 1.
		self.viewer = None

		high = np.array([1., 1., self.max_speed], dtype=np.float32)
		self.action_space = spaces.Box(
			low=-self.max_torque,
			high=self.max_torque, shape=(1,),
			dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-high,
			high=high,
			dtype=np.float32
		)

		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, u):
		th, thdot = self.state  # th := theta

		g = self.g
		m = self.m
		l = self.l
		dt = self.dt

		u = np.clip(u, -self.max_torque, self.max_torque)[0]
		self.last_u = u  # for rendering
		costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

		newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
		newth = th + newthdot * dt
		newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

		self.state = np.array([newth, newthdot])
		return self._get_obs(), -costs, False, {}

	def reset(self):
		high = np.array([np.pi, 1])

		# print(high)

		self.state = self.np_random.uniform(low=-high, high=high)
		self.last_u = None
		return self._get_obs()

	def _get_obs(self):
		theta, thetadot = self.state
		obs = np.array([np.cos(theta), np.sin(theta), thetadot])
		print(obs)
		return obs

	def render(self, mode='human'):
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(500, 500)
			self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

			self.angle_label = pyglet.text.Label('0000', font_size=20, x=10 * 2.2 + 40, y=500 * 2.2 + 40,
												 anchor_x='left', anchor_y='center', color=(0, 255, 0, 255))
			self.viewer.add_geom(DrawText(self.angle_label))

			rod = rendering.make_capsule(1, .2)
			rod.set_color(.8, .3, .3)

			self.pole_transform = rendering.Transform()
			rod.add_attr(self.pole_transform)
			self.viewer.add_geom(rod)
			axle = rendering.make_circle(.05)
			axle.set_color(0, 0, 0)
			self.viewer.add_geom(axle)

			fname = path.join(path.dirname(__file__), "assets/clockwise.png")
			self.img = rendering.Image(fname, 1., 1.)
			self.imgtrans = rendering.Transform()
			self.img.add_attr(self.imgtrans)

		self.viewer.add_onetime(self.img)
		# self.pole_transform.set_rotation( 90 + np.pi / 2 )
		# import time
		# for deg in range(360):
		#     print(deg)
		self.deg = 0
		# self.deg = self.deg / 180 * np.pi
		# self.deg = self.deg + (np.pi / 2)
		# self.pole_transform.set_rotation(np.deg2rad(self.deg) + (np.pi / 2))
		# self.deg = self.deg + 1
		# self.deg = self.deg%360

		# print(self.deg)
		if self.last_u:
			self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

		self.angle_label.text = "10"

		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None


def angle_normalize(x):
	return (((x + np.pi) % (2 * np.pi)) - np.pi)


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import serial, sys, time, re, os
import threading
import logging
import time, struct, math
from scipy.interpolate import interp1d
from numpy import interp
import pyglet
from pyglet import gl


def readline(a_serial, eol=b'\r\n'):
	leneol = len(eol)
	line = bytearray()
	while True:
		c = a_serial.read(1)
		# print(c)
		if c:
			line += c
			if line[-leneol:] == eol:
				break
		else:
			break
	return (line)


# xmap = interp1d([-2.4, 2.4], [0, 0.035])

class DrawText:
	def __init__(self, label: pyglet.text.Label):
		self.label = label

	def render(self):
		self.label.draw()


class NatEnv(gym.Env):
	"""
	Description:
		A pole is attached by an un-actuated joint to a cart, which moves along
		a frictionless track. The pendulum starts upright, and the goal is to
		prevent it from falling over by increasing and reducing the cart's
		velocity.

	Source:
		This environment corresponds to the version of the cart-pole problem
		described by Barto, Sutton, and Anderson

	Observation:
		Type: Box(4)
		Num   Observation               Min             Max
		0     Cart Position             -4.8            4.8
		1     Cart Velocity             -Inf            Inf
		2     Pole Angle                -24 deg         24 deg
		3     Pole Velocity At Tip      -Inf            Inf

	Actions:
		Type: Discrete(2)
		Num   Action
		0     Push cart to the left
		1     Push cart to the right

		Note: The amount the velocity that is reduced or increased is not
		fixed; it depends on the angle the pole is pointing. This is because
		the center of gravity of the pole increases the amount of energy needed
		to move the cart underneath it

	Reward:
		Reward is 1 for every step taken, including the termination step

	Starting State:
		All observations are assigned a uniform random value in [-0.05..0.05]

	Episode Termination:
		Pole Angle is more than 12 degrees.
		Cart Position is more than 2.4 (center of the cart reaches the edge of
		the display).
		Episode length is greater than 200.
		Solved Requirements:
		Considered solved when the average reward is greater than or equal to
		195.0 over 100 consecutive trials.
	"""

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 50
	}

	def read_from_port(self, ser):
		while True:
			try:
				line = readline(ser)
				pendulum_angle, pendulum_velocity, cart_position, cart_velocity, cart_acceleration, limit_A, limit_B = line.decode(
					"utf-8").strip().split(",")

				self.ready = True
			except Exception as e:
				print('exception', e)
			except KeyboardInterrupt:
				print("closing serial port...")
				ser.close()
				sys.exit()
			finally:
				pass

	def control(self, step):
		# print("do control", step)
		v = 400
		global ser
		if step == 0:
			step = -v
		else:
			step = v
		command = [0xff, 0x02]
		command += list(struct.pack(">h", step))

		data_sum = 0
		for x in command:
			data_sum += x

		our_checksum = (~data_sum & 0xFF)
		command.append(our_checksum)
		self.ser.write(command)

	def __init__(self):
		print("init", pendulum)

		def cb(val):
			pendulum_angle, pendulum_velocity, cart_position, cart_velocity, cart_acceleration, limit_A, limit_B = val
			self.ready = True
			limit_A = bool(int(limit_A))
			limit_B = bool(int(limit_B))
			pendulum_angle = float(pendulum_angle)
			pendulum_velocity = float(pendulum_velocity)
			cart_position = float(cart_position)
			cart_velocity = float(cart_velocity)
			cart_acceleration = float(cart_acceleration)

			cart_position = interp(float(cart_position), [0, 0.35], [-2.4, 2.4])
			raw_pendulum_angle = (float(pendulum_angle))
			pendulum_angle = (float(pendulum_angle)) % 360

			if pendulum_angle < 0:
				pendulum_angle = 180.0 + pendulum_angle

			pendulum_angle = interp(pendulum_angle, [0, 360], [0, 360])
			calc_pendulum_angle = interp(pendulum_angle, [0, 360], [-180, 180])

			self.state = (
				calc_pendulum_angle, pendulum_velocity, cart_position, cart_velocity, cart_acceleration, limit_A,
				limit_B)

			self.status = (
				pendulum_angle, pendulum_velocity, cart_position, cart_velocity, cart_acceleration, limit_A,
				limit_B)

			print(self.status)

		ser = pendulum.create('/dev/tty.usbserial-14340')
		pendulum.add_callback(cb)

		self.ser = ser
		self.ready = False

		while not self.ready:
			print("WAIT..")
			time.sleep(0.2)

		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.1
		self.total_mass = (self.masspole + self.masscart)
		self.length = 0.5  # actually half the pole's length
		self.polemass_length = (self.masspole * self.length)
		self.force_mag = 10.0
		self.tau = 0.02  # seconds between state updates
		self.kinematics_integrator = 'euler'

		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4

		# Angle limit set to 2 * theta_threshold_radians so failing observation
		# is still within bounds.
		high = np.array([self.x_threshold * 2,
						 np.finfo(np.float32).max,
						 self.theta_threshold_radians * 2,
						 np.finfo(np.float32).max],
						dtype=np.float32)

		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)

		self.seed()
		self.viewer = None
		self.state = None
		self.status = None

		self.steps_beyond_done = None

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		err_msg = "%r (%s) invalid" % (action, type(action))
		assert self.action_space.contains(action), err_msg
		self.control(action)
		# (pendulum_angle, pendulum_velocity, cart_position, cart_velocity, cart_acceleration, limit_A, limit_B)
		theta, pendulum_velocity, cart_position, cart_velocity, cart_acceleration, limit_A, limit_B = self.state
		# x = cart_position
		# theta = pendulum_angle
		# x, x_dot, theta, theta_dot = self.state

		# self.state = (x, x_dot, theta, theta_dot)
		# print(self.state)
		# costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
		# print(theta)
		done = bool(self.state[-1] or self.state[-2])
		# done = bool(self.state[-1]) or bool(self.state[-2])
		# print(self.state[-1], self.state[-2], done)
		# done = bool(
		#     x < -self.x_threshold
		#     or x > self.x_threshold
		#     or theta < -self.theta_threshold_radians
		#     or theta > self.theta_threshold_radians
		# )

		# done = False
		# costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
		reward = 0
		# if not done:
		reward = -(theta ** 2)
		# elif self.steps_beyond_done is None:
		# 	# Pole just fell!
		# 	self.steps_beyond_done = 0
		# 	reward = 1.0
		# else:
		# 	if self.steps_beyond_done == 0:
		# 		logger.warn(
		# 			"You are calling 'step()' even though this "
		# 			"environment has already returned done = True. You "
		# 			"should always call 'reset()' once you receive 'done = "
		# 			"True' -- any further steps are undefined behavior."
		# 		)
		# 	self.steps_beyond_done += 1
		# 	reward = 0.0

		return np.array(self.state), reward, done, {}

	def _get_obs(self):
		theta, thetadot = self.state
		return np.array([np.cos(theta), np.sin(theta), thetadot])

	def reset(self):
		# self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		self.steps_beyond_done = None
		return np.array(self.state)

	def render(self, mode='human'):
		screen_width = 600 * 2
		screen_height = 400 * 2

		world_width = self.x_threshold * 2
		scale = screen_width / world_width
		carty = 300  # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0
		# position
		# 0
		# 0.350
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			# self.x_label = pyglet.text.Label('0000', font_size=12,
			#     x=10, y=200, anchor_x='left', anchor_y='center', color=(0,0,255,255))
			self.viewer = rendering.Viewer(screen_width, screen_height)

			self.x_label = pyglet.text.Label('0000', font_size=36,
											 x=20, y=400, anchor_x='left', anchor_y='center',
											 color=(255, 0, 0, 255))

			self.angle_label = pyglet.text.Label('0000', font_size=36,
												 x=20, y=450, anchor_x='left', anchor_y='center',
												 color=(0, 255, 0, 255))

			self.episode_label = pyglet.text.Label('0000', font_size=36,
												   x=20, y=500, anchor_x='left', anchor_y='center',
												   color=(0, 255, 0, 255))

			self.viewer.add_geom(DrawText(self.x_label))
			self.viewer.add_geom(DrawText(self.angle_label))
			self.viewer.add_geom(DrawText(self.episode_label))

			l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
			axleoffset = cartheight / 4.0
			cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
			pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
			pole.set_color(.8, .0, .0)

			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)

			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth / 2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5, .5, .8)
			self.viewer.add_geom(self.axle)

			self.track = rendering.Line((0, carty), (screen_width, carty))
			self.track.set_color(0, 0, 0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
		pole.v = [(l, b), (l, t), (r, t), (r, b)]

		# x = self.state
		pendulum_angle, pendulum_velocity, cart_position, cart_velocity, cart_acceleration, limit_A, limit_B = self.status

		# x, x_dot, theta, theta_dot
		self.x_label.text = 'x={}'.format(str(cart_position))
		self.angle_label.text = 'angle={0:.2f}'.format(self.state[0])
		# self.episode_label.text = 'ep={}'.format(str('ep'))

		cartx = cart_position * scale + screen_width / 2.0  # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(np.deg2rad(pendulum_angle) + np.pi)

		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def episode(self, ep, step, reward):
		# print(ep)
		# print(dir(self), ep)
		# print(self.episode_label)
		# if self.episode_label:
		self.episode_label.text = 'ep={}/{} reward={}'.format(str(ep), str(step), reward)

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
