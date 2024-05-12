from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):
    legalMovesG = gameState.getLegalActions()

    def recurse(gameState, gameDepth, agentIndex):
      pass

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):
    legalMovesG = gameState.getLegalActions()

    def recurse(gameState, gameDepth, agentIndex, alpha, beta):
      pass

######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Агент, що реалізує алгоритм Expectimax для випадкових привидів.
    """
    def expectedValue(self, gameState, agentIndex, depth):
        """
        Обчислює очікувану корисність для привида, який діє випадково.
        """
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        values = []
        for successor in successors:
            if successor.isWin() or successor.isLose() or depth == self.depth:
                values.append(self.evaluationFunction(successor))
            else:
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                if nextAgent == 0:
                    values.append(self.maxValue(successor, nextAgent, depth + 1))
                else:
                    values.append(self.expectedValue(successor, nextAgent, depth))

        if not values:
            # Немає легальних ходів, повертаємо значення за замовчуванням або викликаємо функцію оцінки
            return self.evaluationFunction(gameState)
        else:
            return sum(values) / len(values)

    def getAction(self, gameState):
        """
        Повертає дію, визначену алгоритмом Expectimax для випадкових привидів.
        """
        actions = gameState.getLegalActions(0)
        values = [self.expectedValue(gameState.generateSuccessor(0, action), 1, 1) for action in actions]
        maxValue = max(values)
        bestActions = [action for action, value in zip(actions, values) if value == maxValue]
        return random.choice(bestActions)

    def maxValue(self, gameState, agentIndex, depth):
        """
        Обчислює максимальну корисність для Pac-Man.
        """
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)
        values = [self.expectedValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in actions]
        return max(values)


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
    Функція оцінки, яка враховує тип агентів-привидів (випадкові або направлені)
    та відстань до них.
    """
    # Отримуємо інформацію про стан гри
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Обчислюємо відстань до найближчої пілюлі
    foodList = newFood.asList()
    distanceToClosestFood = float("inf")
    if len(foodList) > 0:
        distanceToClosestFood = min([util.manhattanDistance(newPos, food) for food in foodList])

    # Обчислюємо очікувану відстань до привидів залежно від їх типу
    randomGhostDistances = []
    directedGhostDistances = []
    for ghostState in newGhostStates:
        ghostPosition = ghostState.getPosition()
        distance = util.manhattanDistance(newPos, ghostPosition)
        if ghostState.isRandom:
            randomGhostDistances.append(distance)
        else:
            directedGhostDistances.append(distance)

    # Обчислюємо середню відстань до випадкових та направлених привидів
    avgRandomGhostDistance = sum(randomGhostDistances) / max(len(randomGhostDistances), 1)
    avgDirectedGhostDistance = sum(directedGhostDistances) / max(len(directedGhostDistances), 1)

    # Обчислюємо рахунок гри
    score = currentGameState.getScore()

    # Визначаємо вагові коефіцієнти для різних факторів
    foodWeight = 10  # Вага відстані до пілюлі
    randomGhostWeight = -10  # Вага відстані до випадкових привидів
    directedGhostWeight = -20  # Вага відстані до направлених привидів
    scoreWeight = 1  # Вага рахунку гри

    # Обчислюємо загальну оцінку стану гри
    evaluation = scoreWeight * score \
                 - foodWeight * distanceToClosestFood \
                 - randomGhostWeight * avgRandomGhostDistance \
                 - directedGhostWeight * avgDirectedGhostDistance

    # Якщо жоден привид не є "наляканим", додаємо штраф за відстань до привидів
    if sum(newScaredTimes) == 0:
        evaluation -= 500 / (avgRandomGhostDistance + avgDirectedGhostDistance + 1)

    return evaluation