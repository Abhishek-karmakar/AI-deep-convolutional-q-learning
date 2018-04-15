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
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32 , kernel_size = 5) #This will detect one featuer and pass to the next convolutional layer.
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32 , kernel_size = 3) 
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64 , kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons(1,80,80), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
        
        #After the convolution we have to flatten the images and we get the vector to input variable. 

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#making the body - Use the body to specify the method how to play the action.
class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
    
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions
        
#make the AI - Assemble the brain and the body
class AI:
    
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        

#part 2 - Implement Deep Convolutional Q-Learning
    #train the AI.
    

