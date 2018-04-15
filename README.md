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

	
	# Making the CNN

	self.convolution1 = nn.Conv2d(in_channels =1, out_channels = 32, kernel_size)

	input - is One Blackand white images. 
	output - is a image with 32 type of detected images. 
	Kernel - will have a 5x5 image size.

	self.convulation2 = nn.Conv2d(in_channels = 1, out_channels = 32 , kernel_size = 5)

    self.convulation3 = nn.Conv2d(in_channels = 1, out_channels = 32 , kernel_size = 5)     
        
    self.fc1 = nn.Linear(in_features = number_neurons, out_features = 40)

    self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)

    This will have a complete CNN whih will have 3 convolutional layers and 2 full connection layer. 

--------------------------------------

create a method to count_neurons

	- we need a 3 step proess. 
	1.> apply convolution to the input images.
	2.> Max pooling on the convoluted images.  
	3.> Activate the neurons on the pooled convoluted images. 
        
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))

    - To propagate in three convolutional layer. 

        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

    return x.data.view(1, -1).size(1) <-- this will create a enormus vector from the images and pass it to the full connection. 

-------------------------------------

create a forward function

        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

    now just need to flatten the whole image. 

    Then we propagate the flatten layer to the hidden layer. 
            x = F.relu(self.fc1(x))

    Then we need to propagate from the hidden layer to the output layer. 


-------------------------------------

Now we will make the Body of the class which will move the player based on the mind created earlier. We will use Softmax function from PyTorch

	The output functions of the brain will be given as the input to the body. 

	It will use the softmax functions. 

	There are 7 possible actions. Hence there is a distribution of 7 probablilities 

	We can use the T (temperature) parameter to configure the actions. The higher the T, the lower the actions.  


----------------------------
Make the AI class which will use the forward function in it to finally join the body and the brain. 

	To do the whole proagation. 
	1.> Receiving the input images. Since we are getting them from a neural network they will be in a special type. So we'll have to format them in a special structure. The Torch structure. 
	2.> Convert the images into a Numpy array 
	3.> coneert the array into the torch tensor. 
	4.> Finally put the torch tensor into the torch variable whih will contain both the Tensor and the Gradient. (for the dynamic Graph)
	5.> once the above steps are taken then the image will be able to enter the neural network and proper propagation of the signals can take place. 

	 input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))


# Training the AI using Eligibility Trace

 We'll be updating the q values after every N Steps. 

	doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
	Here we are passing the black and white image in 80 x 80 format. 

 Make an AI the earlier made Body and Brain by calling the objects of those classes. 

		cnn = CNN(number_actions)
		softmax_body = SoftmaxBody(T = 1.0)
		ai = AI(brain = cnn, body = softmax_body)

setting up the Eligibility trace using 10000 total steps in Eligibility trace.
	
		n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10) #learning is happening every 10 steps

For every 10 steps we want to get the max points in a batch
	    
	    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state]), dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max

