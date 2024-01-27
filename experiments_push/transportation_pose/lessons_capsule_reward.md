# Lessons learned with capsule reward

- Balance the positive and negative rewards.
A penalty too large, will hinder positive progress.
FOr example, to make sure the agent never leaves a perimeter of the object, a small negative reward every time it is away is enough.
A behavior that is easier to learn can have a larger penalty than a harder one. Because the harder with high penalty will make the agent too afraid.

- Avoid suicide
Sometimes the penalty may be too high, making it attractive for the agent to suicide, receiving a smaller penalty.