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
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) #This will detect one featuer and pass to the next convolutional layer.
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) 
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
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

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32))) #so the array has float32 type variables
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()
    
#part 2 - Implement Deep Convolutional Q-Learning
    #train the AI.
# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n #N is the number of actions that we can take in this environment

# building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# setting up the Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10) #learning is happening every 10 steps
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

#n step Q learning is almost like A3C
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

#Making the moving average on 100 steps. 
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)
ma = MA(100) #because we want out moving our average the AI on 100 steps. 

#Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128): #use a common 32 as a batch size, since we are using 10 steps , thats why we are using 128
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    if avg_reward >= 1500:
        print("Congrats, your AI wins")
        break

#Closing the DOOM environment
doom_env.close()