# Motivations for Mapless Navigation

- Tai, Lei, Giuseppe Paolo, and Ming Liu. "Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation." 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017.

"
Mapless navigation: Motion planning aims at navigating robots to the desired target from the current position without colliding with obstacles. For mobile nonholonomic ground robots, traditional methods, like simultaneous localization and mapping (SLAM), handle this problem through the prior obstacle map of the navigation environment [7] based on dense laser range findings. Manually designed features are extracted to localize the robot and build the obstacle map. There are two less addressed issues for this task: (1) the time-consuming building and updating of the obstacle map, and (2) the high dependence on the precise dense laser sensor for the mapping work and the local costmap prediction. It is still a challenge to rapidly generate appropriate navigation behaviors for mobile robots without an obstacle map and based on sparse range information.
Nowadays, low-cost methods, like WiFi localization [8] and visible-light communication [9], provide lightweight solutions for mobile robot localization. Thus, mobile robots are able to get the real-time target position with respect to the robot coordinate frame. And it is really challenging for a motion planner to generate global navigation behaviors with the local observation and the target position information directly without a global obstacle map. Thus, we present a learningbased mapless motion planner. In virtual environments, a nonholonomic differential drive robot was trained to learn how to arrive at the target position with obstacle avoidance through asynchronous deep reinforcement learning.
"
"
In this paper, a mapless motion planner was trained endto-end through continuous control deep-RL from scratch. We revised the state-of-art continuous deep-RL method so that the training and sample collection can be executed in parallel. By taking the 10-dimensional sparse range findings and the target position relative to the mobile robot coordinate frame as input, the proposed motion planner can be directly applied in unseen real environments without fine-tuning, even though it is only trained in a virtual environment. When compared to the low-dimensional map-based motion planner, our approach proved to be more robust to extremely complicated environments.
"

- Marchesini, Enrico, and Alessandro Farinelli. "Discrete deep reinforcement learning for mapless navigation." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.

"
In more detail, we use the TurtleBot31, which is a widely used platform in several previous work focusing on robot navigation [7], [12]. Figure 1 shows our problem architecture and the robotic platform. We consider a sparse 13-dimensional range finder and the target position with respect to the mobile robot coordinate as input for the network. Traditional methods such as SLAM (Simultaneous Localization and Mapping) are based on dense laser range findings. However, the localization and the local cost-map prediction heavily depends on precise dense laser sensor. The laser sensor shipped with the TurtleBot3 is an LDS012 , which has an update rate of maximum 5Hz. This low rate causes planning issues using traditional methods (see Section V) where both the scan values and the localization of the robot have to be precise.
"

- Zhu, Kai, and Tao Zhang. "Deep reinforcement learning based mobile robot navigation: A review." Tsinghua Science and Technology 26.5 (2021): 674-691.

Nothing.

- Sun, Huihui, et al. "Motion planning for mobile robots—Focusing on deep reinforcement learning: A systematic review." IEEE Access 9 (2021): 69061-69081.

Nothing

- Grando, Ricardo Bedin, et al. "Double critic deep reinforcement learning for mapless 3d navigation of unmanned aerial vehicles." Journal of Intelligent & Robotic Systems 104.2 (2022): 29.

Nothing.

- Hamza, Ameer. "Deep Reinforcement Learning for Mapless Mobile Robot Navigation." (2022).
"
Conventional navigation approaches depend on predefined accurate obstacle maps and costly high-end precise laser sensors. These maps are difficult and expensive to acquire and degrade due changes in the environment. This limits the overall use of mobile robots in dynamic settings. 
"
"
Tradition robot navigation system consists of three main components: mapping, localization, and path planning as shown in Figure 1. The global map of the unknow environment is constructed either by the mapping system, mostly using Simultaneous Localization and Mapping (SLAM), or manually designed by humans using ranging and visual sensors data, (Ruan et al., 2019; Zhu, K. & Zhang, T., 2021). The path planning module contains global and local planner. The global path planner proposes waypoints to local planner for optimal trajectory to carryout navigation tasks like reaching a target while avoiding obstacles. Planning module is dependent on accurate map, good localization / position estimation, sensor data, and target position to determine optimal path 
"
"
The construction and maintenance of the map of the environment is computationally expensive and require dense precise laser sensors (e.g., in SLAM) if done dynamically and is sensitive to sensor noise (Zeng, 2018).
"
"
However, there are significant challenges to be addressed while using SLAM for complex tasks. Mobile robots have limited computation power and computations are performed on low energy embed microprocessors. For accurate mapping and localization high dimensional data from the lasers or cameras needs to be processed at a high frequency with limited computation resources.
"
"
Real environments have numerous movable objects or dynamic conditions e.g., light which directly affects the static environment assumption of most SLAM algorithms. These changes can result in conflict while updating the map and may result in inconsistent map, whereas overall map building is a time-consuming process
"
"
Moreover, the performance of these traditional approach is constrained by the accuracy of the grid map representation and its storage requirements and the intensive computation for real time path replanning in dynamic environment (Zhu & Zhang, 2021).
"
"
Traditional navigation framework has different critical components which merit research of their own and integration of these component aggregate and magnify the computational errors which leads to poor performance 
"
"
Mapless approach relieves the requirement of global map information hence performance of navigation system is no longer depended on the quality of the global map. Moreover, this approach models sensor information and relative destination position directly to robot actions mainly using neural networks (as shown in Figure 2) and has a strong learning ability with low dependence on sensor accuracy 
"
"
Two approaches: LiDAR based and vision-based approaches.
"
"
Hence, mapless approach is commonly used for tasks in which there is no explicit destination such as obstacle avoidance or a known destination is given the local coordinate frame of the robot such as local navigation (Xie, 2020).
"

- Tsai, Chi-Yi, Humaira Nisar, and Yu-Chen Hu. "Mapless LiDAR navigation control of wheeled mobile robots based on deep imitation learning." IEEE Access 9 (2021): 117527-117541

"
To simplify the architecture of the traditional map-based navigation control system, the development of mapless LiDAR navigation technology for wheeled mobile robots has received more and more attention in recent years. Figure 1(b) shows the system architecture of the existing deep learning-based mapless LiDAR navigation control system, which employs the neural network model to predict the motion control commands for the robot to approach the target while avoiding obstacles. Compared with traditional map-based navigation technology, the existing deep learning-based mapless navigation technology has two major advantages:

It releases the requirement of global map information. This advantage makes the performance of the navigation system independent of the quality of the global map, but it depends on the diversity of the training dataset.

It directly uses the sensor information and the relative target position to predict the corresponding motion control commands. This advantage simplifies the complexity of the navigation system by using the neural network model. However, a potential flaw of this simplification is that the user may need to retrain the entire neural network model if the hardware specification of the mobile platform is changed.
"