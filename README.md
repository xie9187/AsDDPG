# Learning with training wheels: Speeding up training with a simple controller for Deep Reinforcement Learning

By [Linhai Xie](https://www.cs.ox.ac.uk/people/linhai.xie/), [Sen Wang](http://senwang.weebly.com/), Stefano Rosa,  Niki trigoni, Andrew Markham.

The tensorflow implmentation for the paper: [Learning with training wheels: Speeding up training with a simple controller for Deep Reinforcement Learning](http://www.cs.ox.ac.uk/files/9953/Learning%20with%20Training%20Wheels.pdf)

## Contents
0. [Introduction](#Introduction)
0. [Prerequisite](#Prerequisite)
0. [Instruction](#instruction)
0. [Citation](#citation)

## Introduction

In this project we proposed a switching machanism to let the agent learn from another simple controller, e.g. PID, during training instead of purely random exploration and speed up the training of DDPG.

For details please see the [paper](https://www.cs.ox.ac.uk/files/9953/Learning%20with%20Training%20Wheels.pdf)


The implementation of DDPG is based on [Emami's work](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html).


## Prerequisites

Tensorflow > 1.1

ROS Kinetic

ros stage

matplotlib

cv2

## Instruction

roscore

rosrun stage_ros stageros PATH TO THE FOLDER/AsDDPG/worlds/Obstacles.world

python DDPG.py

## Citation

If you use this method in your research, please cite:

	@inproceedings{xie2017Learning,
  		title = "Learning with Training Wheels: Speeding up Training with a Simple Controller for Deep Reinforcement Learning",
  		author = "Xie, Linhai and Wang, Sen and Rosa, Stefano and Markham, Andrew and Trigoni, Niki",
  		year = "2018",
  		booktitle = "Robotics and Automation (ICRA), 2017 IEEE/RSJ International Conference on",
  		note = "accepted",
  		organization = "IEEE",
	}



