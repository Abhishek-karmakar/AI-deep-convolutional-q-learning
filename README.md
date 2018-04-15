# Deep Convolutional Q Learning

# Overview
	
	Deep Convolutional Q-Learning Intuition.
	Eligibility Trace (N-Step Q-Learning)


# Deep Convolutional Q Learning 
	
	In Q Learning the Agent has the information and it's primarily required to transfer that to a input-vector.

	Its too simple for AI to get the data in form of a vector. Humans learn from sight. 

	So the environment is looked at as an image. This image is passed to the Convolutional Layer for various operations (look details in CNN) and then sent to pooling(check CNN for more details). Then this information is passed to the input later. 

	Then this information will be passed to the input layer and it can take many actions. Turn Left, Turn Right, Shoot, Jump etc.

# Eligibility Trace - N-Step Q-Learning

	In case of Q Learning an agent takes step based on the updated wiughts at every step. The only compass is the reward that its getting. 

	In Eligibility Trace the agent takes n-steps and evaluates them again to check if the step taken was correct or not. It has a out of the next N steps. 

	The agent is not only looking at the comulative rewards to take its decision but it also has a trace that is kept in algorithm which checks that if the agent gets a negative reward then those steps should get updated. 

# Installations (for linux)

	1.> install OpenAI
		git clone https://github.com/openai/gym
		cd gym
		pip install -e .

	2.> install other dependencies.
		sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

	3.> Install ppaquette to visualize Doom
		pip install ppaquette-gym-doom

	4.> Install ffmpeg to get the videos in a folder
		conda install -c conda-forge ffmpeg=3.2.4

	5.> install vizdoom for the visual doom envirnment. 
		pip install vizdoom

	6.> clone and install doom-py
		git clone https://github.com/open-ai/doom-py 
		pip install -e .


# Coding the DOOM AI

	The visual page is not present upfont but can be seen under this link 
		https://gym.openai.com/envs/DoomCorridor-v0/