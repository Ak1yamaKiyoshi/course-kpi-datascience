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


# 3. Define the Value Iteration algorithm
def value_iteration(env, gamma=1.0, theta=1e-8):
    """
    Value Iteration Algorithm to find the optimal policy
    
    Args:
        env: OpenAI gym environment
        gamma: Discount factor
        theta: Stopping criterion for value iteration
    Returns:
        policy: Optimal policy
        V: Value function
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=np.int32)
    
    while True:
        delta = 0
        for s in range(nS):
            old_v = V[s]
            V[s] = max(sum(p * (r + gamma * V[s_prime])
                           for p, s_prime, r, done in env.P[s][a])
                       for a in range(nA))
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
        
    for s in range(nS):
        policy[s] = max(
            (sum(p * (r + gamma * V[s_prime])
                 for p, s_prime, r, done in env.P[s][a]), a)
            for a in range(nA))[1]
        
    return policy, V



# 4. Train the agent using Value Iteration
policy, V = value_iteration(env)

# 5. Visualize the trained policy
state = env.reset()
state = state[0]  # Extract the state integer from the tuple
env_image = env.render()
plt.figure()
plt.imshow(env_image)
plt.show()
print(policy.reshape(4, 4))



input("> Press Something to see first iteration")
state = env.reset()
state = state[0]
done = False
while not done:
    action = policy[state]
    state, reward, done, truncated, info = env.step(action)
    state = state  # 
    env_image = env.render()
    
    cv2.imshow("",cv2.cvtColor(env_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(110)
    print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")


# 6. Test the trained agent
episodes = 1_155_000*10
print(f"Initial State: {state}")
for cur_episode in range(episodes):
    state = env.reset()
    state = state[0]
    
    if cur_episode % 1000 == 0:
        print(f"cur episode: {cur_episode}")


toexit = False
while not toexit:
    if "q" in input("> Press Something to see last iteration or q to exit"):
        toexit = True
    state = env.reset()
    state = state[0]
    done = False
    while not done:
        action = policy[state]
        state, reward, done, truncated, info = env.step(action)
        state = state  # 
        env_image = env.render()
        
        cv2.imshow("",cv2.cvtColor(env_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(110)
        print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
