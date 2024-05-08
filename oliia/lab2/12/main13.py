import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.toy_text import frozen_lake
import cv2

# 1. Import required libraries
import gym
from gym.envs.toy_text import frozen_lake

# 2. Create the FrozenLake environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")

# 3. Define the Q-learning algorithm
def q_learning(env, num_episodes=1000, alpha=0.5, gamma=0.99, epsilon=0.2):
    """
    Q-learning algorithm to find the optimal policy

    Args:
        env: OpenAI gym environment
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Q: Q-value function
        policy: Optimal policy
    """
    nS = env.observation_space.n
    nA = env.action_space.n

    Q = np.zeros((nS, nA))
    policy = np.zeros(nS, dtype=np.int32)

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False

        if episode % 1000 == 0:
            print(f"episode: {episode}")

        while not done:
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done, truncated, _ = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state

    for s in range(nS):
        policy[s] = np.argmax(Q[s])

    return Q, policy


Q, policy = q_learning(env, num_episodes=10)

state = env.reset()
state = state[0]  
print(policy.reshape(4, 4))

input("> Press Something to see first iteration")
state = env.reset()
state = state[0]
done = False
while not done:
    action = policy[state]
    state, reward, done, truncated, info = env.step(action)
    state = state  
    env_image = env.render()
    
    cv2.imshow("",cv2.cvtColor(env_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(110)
    print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

Q, policy = q_learning(env, num_episodes=300_000)

input("> Press Something to see first iteration")
state = env.reset()
state = state[0]
done = False
while not done:
    action = policy[state]
    state, reward, done, truncated, info = env.step(action)
    state = state  
    env_image = env.render()
    
    cv2.imshow("",cv2.cvtColor(env_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(110)
    print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

