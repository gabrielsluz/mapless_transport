# Lessons learned with capsule reward

- Balance the positive and negative rewards.
A penalty too large, will hinder positive progress.
FOr example, to make sure the agent never leaves a perimeter of the object, a small negative reward every time it is away is enough.
A behavior that is easier to learn can have a larger penalty than a harder one. Because the harder with high penalty will make the agent too afraid.

- Avoid suicide
Sometimes the penalty may be too high, making it attractive for the agent to suicide, receiving a smaller penalty.

- Success Reward is important
It is the long term goal. It should be balanced to make the long term goal important.

- Keep the agent inside the radius of the object because it really helps training
No need to terminate the episode or to give penalties.
Though, it speeds up training in the start and middle.

- Potential Reward Shaping is guidance, but not good for capsule, because the final reward is unchanged.

- Orient and distance progress rewards should be potential funcions, but they should be scaled differently.
If the agent should start worrying about orientation from dist 10, and dist is being normalized by 25, then in the last 10 meters orientation reward will be disproportionally bigger.
It should be scaled to 0.4 times the value.