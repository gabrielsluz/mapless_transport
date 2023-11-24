# Understanding the problem

Mapless transportation can be divided into: 
- Decide where the object should go => Navigation
- Control the robot to push the object => transportation

Task: Understand approaches for local navigation and planar pushing, in order to better design the RL solution.

## Navigation

Main questions I want to answer:
- What is the problem of mapless navigation?
    - Assumptions
    - Static x dynamic obstacles
    - How the community calls it?
- What is the state-of-the-art?
- How RL has been applied?

### VECTOR FIELD HISTOGRAM (VFH)
https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/borenstein_VFHisto.pdf 

Is the grid a global representation? Why no use SLAM then?

Window: 33x33, always centered about the robot.

### Examples in RL
Get examples of how researches have solved navigaiton in RL.

## Pushing
Stable Pushing: Mechanics, Controllability, and Planning

## Dimensions
Set up a realistic scenario based on a real robot and real sensors.
If I wanted to implement it on a real robot, how?
Example: HeRo or Pioneer. 
- How to control the robot? => holonomic? Differential?
- Sizes of the obstacles?
- How tight should the empty corridors be?
