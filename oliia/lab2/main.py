import gym
from gym import spaces
import numpy as np
import cv2

class RoomWithObstacles(gym.Env):
    def __init__(self, room_size=(10, 10), obstacles=[(3, 3), (3, 6), (6, 6), (8, 3)]):
        self.room_size = room_size
        self.obstacles = obstacles
        self.observation_space = spaces.Box(low=0, high=2, shape=room_size, dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.transition_model = {
            0: lambda state, row, col: (max(row - 1, 0), col),
            1: lambda state, row, col: (min(row + 1, self.room_size[0] - 1), col),
            2: lambda state, row, col: (row, max(col - 1, 0)),
            3: lambda state, row, col: (row, min(col + 1, self.room_size[1] - 1))
        }

        self.reset()

    def reset(self):
        self.agent_position = (0, 0)
        self.state = np.zeros(self.room_size, dtype=np.int32)
        for obstacle in self.obstacles:
            self.state[obstacle] = 1
        self.state[self.agent_position] = 2
        return self.state

    def step(self, action):
        row, col = self.agent_position
        new_row, new_col = self.transition_model[action](self.state, row, col)
        new_position = (new_row, new_col)

        if new_position in self.obstacles:
            reward = -1.0
            done = False
        else:
            self.agent_position = new_position
            self.state[row, col] = 0
            self.state[new_row, new_col] = 2
            reward = -0.1
            done = False

        return self.state, reward, done, {}

    def render(self, mode='human'):
        image = np.zeros((self.room_size[0], self.room_size[1], 3), dtype=np.uint8)

        for obstacle in self.obstacles:
            image[obstacle] = (0, 0, 0)

        image[self.agent_position] = (0, 255, 0)

        cv2.imshow("Environment", image)
        cv2.waitKey(1)

def value_iteration(env, gamma, theta=1e-8):
    V = np.zeros(env.observation_space.shape)
    policy = np.zeros(env.observation_space.shape, dtype=np.int32)

    while True:
        delta = 0
        for s in range(env.observation_space.shape[0] * env.observation_space.shape[1]):
            row, col = np.unravel_index(s, env.observation_space.shape)
            if (row, col) not in env.obstacles:
                v = V[row, col]
                action_values = []
                for a in range(env.action_space.n):
                    new_row, new_col = env.transition_model[a](env.state, row, col)
                    new_position = (new_row, new_col)
                    if new_position in env.obstacles:
                        reward = -1.0
                    else:
                        reward = -0.1
                    action_values.append(reward + gamma * V[new_row, new_col])
                V[row, col] = max(action_values)
                delta = max(delta, abs(v - V[row, col]))
        if delta < theta:
            break

    for s in range(env.observation_space.shape[0] * env.observation_space.shape[1]):
        row, col = np.unravel_index(s, env.observation_space.shape)
        if (row, col) not in env.obstacles:
            action_values = []
            for a in range(env.action_space.n):
                new_row, new_col = env.transition_model[a](env.state, row, col)
                new_position = (new_row, new_col)
                if new_position in env.obstacles:
                    reward = -1.0
                else:
                    reward = -0.1
                action_values.append(reward + gamma * V[new_row, new_col])
            policy[row, col] = np.argmax(np.array(action_values))

    return V, policy

def visualize_environment(env, policy):
    image = np.zeros((env.room_size[0], env.room_size[1], 3), dtype=np.uint8)

    for obstacle in env.obstacles:
        image[obstacle] = (0, 0, 0)

    for i in range(env.room_size[0]):
        for j in range(env.room_size[1]):
            if (i, j) not in env.obstacles:
                action = policy[i, j]
                if action == 0:  # Up
                    image[i, j] = (0, 255, 0)
                    cv2.arrowedLine(image, (j, i), (j, i - 1), (0, 0, 255), 1, cv2.LINE_AA)
                elif action == 1:  # Down
                    image[i, j] = (0, 255, 0)
                    cv2.arrowedLine(image, (j, i), (j, i + 1), (0, 0, 255), 1, cv2.LINE_AA)
                elif action == 2:  # Left
                    image[i, j] = (0, 255, 0)
                    cv2.arrowedLine(image, (j, i), (j - 1, i), (0, 0, 255), 1, cv2.LINE_AA)
                else:  # Right
                    image[i, j] = (0, 255, 0)
                    cv2.arrowedLine(image, (j, i), (j + 1, i), (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Environment", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

env = RoomWithObstacles()
gamma = 0.9  # Discount factor
V, policy = value_iteration(env, gamma)
visualize_environment(env, policy)