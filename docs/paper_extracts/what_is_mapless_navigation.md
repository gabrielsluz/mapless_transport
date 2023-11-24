# What is Mapless Navigation?

- Tai, Lei, Giuseppe Paolo, and Ming Liu. "Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation." 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017.

"
Abstract— We present a learning-based mapless motion planner by taking the sparse 10-dimensional range findings and the target position with respect to the mobile robot coordinate frame as input and the continuous steering commands as output. Traditional motion planners for mobile ground robots with a laser range sensor mostly depend on the obstacle map of the navigation environment where both the highly precise laser sensor and the obstacle map building work of the environment are indispensable. We show that, through an asynchronous deep reinforcement learning method, a mapless motion planner can be trained end-to-end without any manually designed features and prior demonstrations. The trained planner can be directly applied in unseen virtual and real environments. The experiments show that the proposed mapless motion planner can navigate the nonholonomic mobile robot to the desired targets without colliding with any obstacles.
"

- Marchesini, Enrico, and Alessandro Farinelli. "Discrete deep reinforcement learning for mapless navigation." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.

"
Abstract— Our goal is to investigate whether discrete state space algorithms are a viable solution to continuous alternatives for mapless navigation. To this end we present an approach based on Double Deep Q-Network and employ parallel asynchronous training and a multi-batch Priority Experience Replay to reduce the training time. Experiments show that our method trains faster and outperforms both the continuous Deep Deterministic Policy Gradient and Proximal Policy Optimization algorithms. Moreover, we train the models in a custom environment built on the recent Unity learning toolkit and show that they can be exported on the TurtleBot3 simulator and to the real robot without further training. Overall our optimized method is 40% faster compared to the original discrete algorithm. This setting significantly reduces the training times with respect to the continuous algorithms, maintaining a similar level of success rate hence being a viable alternative for mapless navigation.
"
"
In this paper, we focus on the mapless navigation problem, a well-known benchmark in recent DRL literature [7], [8], which aims at navigating the robot towards a random target using local observation and the target position, without a map of the surrounding environment or obstacles.
"

- Zhu, Kai, and Tao Zhang. "Deep reinforcement learning based mobile robot navigation: A review." Tsinghua Science and Technology 26.5 (2021): 674-691.

"
In the past five years, several studies have been conducted on DRL-based navigation, but the classification of DRL-based navigation remains confusing. For example, when using lightweight localization solutions, such as GPS and Wifi, a DRLbased navigation system can obtain the relative position of a goal point without global map information, which several researchers refer to as “mapless” navigation. In other research, the DRL method preprocesses the sensor’s local observation data into the form of a local map, which is called a “map-based” method, and global map information is not used. Moreover, some studies refer to “visual navigation” as the use of a first-person-view Red-Green-Blue (RGB) image as the target, whereas other studies refer to navigation based on visual sensors.
"

- Sun, Huihui, et al. "Motion planning for mobile robots—Focusing on deep reinforcement learning: A systematic review." IEEE Access 9 (2021): 69061-69081.

Nothing

- Grando, Ricardo Bedin, et al. "Double critic deep reinforcement learning for mapless 3d navigation of unmanned aerial vehicles." Journal of Intelligent & Robotic Systems 104.2 (2022): 29.

"
Abstract This paper presents a novel deep reinforcement learning-based system for 3D mapless navigation for Unmanned Aerial Vehicles (UAVs). Instead of using a image-based sensing approach, we propose a simple learning system that uses only a few sparse range data from a distance sensor to train a learning agent. We based our approaches on two state-of-art double critic Deep-RL models: Twin Delayed Deep Deterministic Policy Gradient (TD3) and Soft Actor-Critic (SAC). We show that our two approaches manage to outperform an approach based on the Deep Deterministic Policy Gradient (DDPG) technique and the BUG2 algorithm. Also, our new Deep-RL structure based on Recurrent Neural Networks (RNNs) outperforms the current structure used to perform mapless navigation of mobile robots. Overall, we conclude that Deep-RL approaches based on double critic with Recurrent Neural Networks (RNNs) are better suited to perform mapless navigation and obstacle avoidance of UAVs
"
"
In our previous work, we proposed the adaptation of two Deep-RL techniques for an UAV in two environments [12] and explored the concept for mobile robots [13]. Based on the state-of-art structure for terrestrial mobile robots [7], we investigated the performance of Deep-RL for 2D mapless navigation related tasks of UAVs. However, we propose a new structure for the agents for 3D navigation and develop a new deterministic approach based on the Twin Delayed Deep Deterministic Policy Gradient (TD3) [14] extending our previous work. Instead of using the conventional structure used to perform mapless navigation of terrestrial mobile robots, we propose a new methodology with Recurrent Neural Networks (RNNs) with more sophisticated Deep-RL approaches.
"
"
The objective of this work is to show how well Deep-RL approaches based on RNNs can perform in 3D navigation-related tasks. In this work we also include environmental difficulties, such as simulated wind, to improve the overall robustness. Our approach was done in such a way that the network used had only 26 inputs and 3 outputs, as shown in Figure 1. As inputs, it was used readings of a distance sensor coupled to the upper part of the vehicle, the previous linear velocities, the delta yaw angle and the distance and angle of the UAV related to the target. The actions were set as the outputs of the network to send it to the vehicle to arrive at the target. We expect that the intelligent agent will not just be able to arrive at the target goal but also be able to avoid any possible collision on the way to get it. We also expect the agents to perform through the environmental difficulties, showing robustness on the adaptation. Our improved deterministic approach (3DDC-D) and our bias stochastic approach (3DDC-S) are set to be compared with our previous Deep-RL approaches and with a behavior-based approach inspired in the work of [15] that can be used to perform mapless navigation of UAVs.
"
"
Differently, our method addresses 3D mapless navigation and waypoint mapless navigation tasks by relying solely on laser sensor readings and the vehicle’s relative localization data.
"

- Hamza, Ameer. "Deep Reinforcement Learning for Mapless Mobile Robot Navigation." (2022).
