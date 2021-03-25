# OpenAI Gym Simple2DSimulator environments

## Requirement 
- python3.6.9
- OpenAI Gym

## Building OpenAI Gym from source code

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

## Environment

```python
import gym
import Simple2dSimulator

env = gym.make('Simple2dSimulator-v0')
env.reset()

for _ i in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done,  _ = env.step(action)

    if done:
        env.reset()
```


