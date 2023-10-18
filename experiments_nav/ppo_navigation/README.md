# Goal: Achieve similar perfomance to potential fields

## Potential Fields:
- Number of rays: 24
- Min dist for Repulsion: 5
- Force length: 0.5
- max_steps: 200


### Environment:
'circle_line', 'small_4_circles',
'4_circles', 'sparse_1', 'sparse_2',
'16_circles', '25_circles', '49_circles',
'1_circle', '1_rectangle', '1_triangle',

Success Rate = 93%

## PPO

- Observation:
    - Laser:
        - n_rays: 24
        - range: 5
        - not detected value: 1.0
    - Goal:
        - Distance and angle
    - Memory: 0

- Agent:
    - n_actions: 8
    - Force length: 0.5

- Reward:
    - max_steps: 300
    - progress_weight: 1
    - time_penalty: -0.2

Success rate = 85%