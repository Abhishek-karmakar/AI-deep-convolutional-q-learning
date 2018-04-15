#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#AI for Doom

"""
Created on Sun Apr 15 20:35:15 2018

@author: abhishek

"""
#importing the libraries
import numpy as np # to work with arrys
import torch #because we are implementing this neural network with pytorch
import torch.nn as nn # neural network, convolutional layers of our neural network
import torch.nn.functional as F 
import torch.optim as optim #all the activation functions for our neural network
from torch.autograd import Variable

#importing the package for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

#importig the other python files
import experience_replay, image_preprocessing

#part 1 - Building the AI
    #making the brain - detects the image an what is the Q-Value
    
class CNN(nn.module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        #first create the convolution cnnection then pass it to neural network to create the full connection
        self.convolution1 #This will detect one featuer and pass to the next convolutional layer.
        self.convolution2 
        self.convolution3
        self.fc1
        self.fc2
        
#After the convolution we have to flatten the images and we get the vector to input variable. 
    
#making the body - Use the body to specify the method how to play the action.
    
#make the AI - Assemble the brain and the body
    

#part 2 - Implement Deep Convolutional Q-Learning
    #train the AI.
    

