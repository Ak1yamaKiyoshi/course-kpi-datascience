import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.toy_text import frozen_lake
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import gym
from gym.envs.toy_text import frozen_lake
 
def value_iteration(env, gamma=1.0, theta=1e-8):
    nS = env.observation_space.n
    nA = env.action_space.n
    
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=np.int32)
    
    while True:
        delta = 0
        for s in range(nS):
            old_v = V[s]
            new_v = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    new_v[a] += prob * (reward + gamma * V[next_state])
            V[s] = np.max(new_v)
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
        
    for s in range(nS):
        new_v = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[s][a]:
                new_v[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(new_v)
        
    return policy, V 


def run_episode(env, policy):
    state = env.reset()[0]
    total_reward = 0
    steps_to_win = 0
    done = False
    while not done:
        action = policy[state]
        state, reward, done, truncated, info = env.step(action)
        state = state
        total_reward += reward
        steps_to_win += 1
    return total_reward, steps_to_win

def run(gamma, theta):
    print(gamma, theta)
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
 
    policy, V = value_iteration(env, gamma, theta)
    rewards_per_episode = []
    steps_to_win_per_episode = []
    episodes = 10_155  

    for cur_episode in range(episodes):
        total_reward, steps_to_win = run_episode(env, policy)
        rewards_per_episode.append(total_reward)
        steps_to_win_per_episode.append(steps_to_win)
        if cur_episode % 100 == 0:
            print(f"Episode: {cur_episode}, Total Reward: {total_reward}, Steps to Win: {steps_to_win}")

    return rewards_per_episode, steps_to_win_per_episode

parameters = [
    (0.1, 1.0),
    (-1.0, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
    (1.0, 0.1),
    (1.0, 0.5),
    (1.0, 1.0),
]
results = {}  

for gamma, theta in parameters:
    rewards, steps = run(gamma, theta)
    results[(gamma, theta)] = (rewards, steps)

plt.figure(figsize=(12, 8))
for param, (rewards, steps) in results.items():
    x = np.array(range(len(steps))).reshape(-1, 1)
    y = np.array(steps).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    plt.plot(x, reg.predict(x), linestyle='--', label=f"gamma: {param[0]}, theta: {param[1]}")
plt.xlabel("Episode")
plt.ylabel("Steps to Win")
plt.title("Steps to Win per Episode for Different Parameter Sets")
plt.legend()
plt.show()


input("> Press Something ")
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
policy, V = value_iteration(env, 1.0, 0.1)

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

