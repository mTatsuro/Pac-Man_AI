# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
from searchAgents import AnyFoodSearchProblem
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    foodOnLeft = sum(int(j) for i in newFood for j in i)

    if foodOnLeft > 0:
        foodDist= [manhattanDistance(newPos, (x, y))
                              for x, row in enumerate(newFood)
                              for y, food in enumerate(row)
                              if food]
        closestFood = min(foodDist)
    else:
        closestFood = 0

    if newGhostStates:
        ghostDist = []
        for ghost in newGhostStates:
            ghostDist.append(manhattanDistance(ghost.getPosition(), newPos))
        closestGhost = min(ghostDist)

        if closestGhost == 0:
                closestGhost = -2000
        else:
                closestGhost = -5 / closestGhost
    else:
        closestGhost = 0

    score = -2 * closestFood + closestGhost - 40 * foodOnLeft
    return score

def scoreEvaluationFunction(currentGameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def minValue(state, agentIndex, depth):
        possibleMoves = state.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()

        if not possibleMoves:
            return self.evaluationFunction(state)

        if agentIndex == numAgents - 1:
            minimum = min(maxValue(state.generateSuccessor(agentIndex, action), agentIndex,  depth) for action in possibleMoves)
        else:
            minimum = min(minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in possibleMoves)

        return minimum

    def maxValue(state, agentIndex, depth):
        agentIndex = 0
        possibleMoves = state.getLegalActions(agentIndex)

        if not possibleMoves or depth == self.depth:
            return self.evaluationFunction(state)

        maximum = max(minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) for action in possibleMoves)

        return maximum

    actions = gameState.getLegalActions(0)

    allActions = {}
    for action in actions:
        allActions[action] = minValue(gameState.generateSuccessor(0, action), 1, 1)

    return max(allActions, key=allActions.get)
    util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def minValue(state, agentIndex, depth, alpha, beta):
        possibleMoves = state.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()

        # sanity check
        if not possibleMoves:
            return self.evaluationFunction(state)

        minimum = 99999
        currentBeta = beta

        if agentIndex == numAgents - 1:
            for action in possibleMoves:
                minimum =  min(minimum, maxValue(state.generateSuccessor(agentIndex, action), agentIndex,  depth, alpha, currentBeta))
                if minimum < alpha:
                    return minimum
                currentBeta = min(currentBeta, minimum)

        else:
            for action in possibleMoves:
                minimum =  min(minimum, minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, currentBeta))
                if minimum < alpha:
                    return minimum
                currentBeta = min(currentBeta, minimum)

        return minimum

    def maxValue(state, agentIndex, depth, alpha, beta):

        agentIndex = 0
        possibleMoves = state.getLegalActions(agentIndex)

        # sanity check
        if not possibleMoves or depth == self.depth:
            return self.evaluationFunction(state)

        minimum = -99999 # arbitrary small number
        currentAlpha = alpha

        for action in possibleMoves:
            minimum = max(minimum, minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1, currentAlpha, beta) )
            if minimum > beta:
                return minimum
            currentAlpha = max(currentAlpha, minimum)
        return minimum

    alpha = -99999
    beta = 99999
    actions = gameState.getLegalActions(0)

    allActions = {}
    for action in actions:
        value = minValue(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
        allActions[action] = value
        if value > beta:
            return action
        alpha = max(value, alpha)

    return max(allActions, key=allActions.get)
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    def expectedValue(state, agentIndex, depth):
        numAgents = gameState.getNumAgents()
        possbileMoves = state.getLegalActions(agentIndex)

        if not possbileMoves:
            return self.evaluationFunction(state)

        expectation = 0
        probabilty = 1.0 / len(possbileMoves)

        for action in possbileMoves:
            if agentIndex == numAgents - 1:
                currentExpValue =  maxValue(state.generateSuccessor(agentIndex, action), agentIndex,  depth)
            else:
                currentExpValue = expectedValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
            expectation += currentExpValue * probabilty

        return expectation

    def maxValue(state, agentIndex, depth):

        agentIndex = 0
        possibleMoves = state.getLegalActions(agentIndex)

        if not possibleMoves or depth == self.depth:
            return self.evaluationFunction(state)

        maximum = max(expectedValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) for action in possibleMoves)

        return maximum

    actions = gameState.getLegalActions(0)
    allActions = {}
    for action in actions:
        allActions[action] = expectedValue(gameState.generateSuccessor(0, action), 1, 1)
    return max(allActions, key=allActions.get)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def closestFoodHeuristic(pos, problem, info={}):
    food = problem.food
    foodDist = [
        util.manhattanDistance(pos, (x, y))
        for x, row in enumerate(food)
        for y, food_bool in enumerate(row)
        if food_bool
        ]

    return min(foodDist) if foodDist else 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    newStates = util.PriorityQueue()
    visitedStates = []

    # get the first
    newStates.push((problem.getStartState(), []), 0)

    # breadth first search
    while not newStates.isEmpty():
        states, actions = newStates.pop()
        if not states in visitedStates:
            visitedStates.append(states)
            if problem.isGoalState(states):
                return actions
            for successor in problem.getSuccessors(states):
                if not successor[0] in visitedStates:
                    cost = problem.getCostOfActions(actions + [successor[1]]) + heuristic(successor[0], problem)
                    newStates.push((successor[0], actions + [successor[1]]), cost)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Each state is evaluated based on the weighted sum of the following variables:
    - the distance to the closest food
    - the distance to the closest scared ghost
    - the distance to the closest power pellet
    - the distance to the closest ghost
    - the number of remaining foods
    - the number of remaining power pellets
    The first three are given positive weights and the latter three are given negative weights.
  """
  "*** YOUR CODE HERE ***"
  currentPosition = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  ghosts = currentGameState.getGhostStates()
  capsules = currentGameState.getCapsules()

  remainingFood = sum(int(j) for i in food for j in i)

  problem = AnyFoodSearchProblem(currentGameState)
  closestFood = aStarSearch(problem, heuristic = closestFoodHeuristic)
  if closestFood:
      closestFood = 1 / len(closestFood)
  else:
      closestFood = 1000

  scared = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
  ghosts = [ghost for ghost in ghosts if ghost.scaredTimer == 0]

  if ghosts:
      ghostDist = []
      for ghost in ghosts:
          ghostDist.append(manhattanDistance(ghost.getPosition(), currentPosition))
      closestGhost = min(ghostDist)

      if closestGhost == 0:
          closestGhost = 200000
      else:
          closestGhost = 1 / closestGhost
  else:
      closestGhost = 0

  closestScared = 0
  if scared:
      scaredDist = []
      for ghost in scared:
          scaredDist.append(manhattanDistance(ghost.getPosition(), currentPosition))
      scaredDist = [distance for ghost, distance in zip(scared, scaredDist)
                             if distance <= ghost.scaredTimer]

      if scaredDist:
          closestScared = min(scaredDist)
          if closestScared == 0:
              closestScared = 10
          else:
              closestScared = 1 / closestScared

  remainingCapsules = len(capsules)
  if capsules:
      capsuleDist = []
      for capsule in capsules:
          capsuleDist.append(manhattanDistance(capsule, currentPosition))
      closestCapsule = 1 / min(capsuleDist)
  else:
      closestCapsule = 0

  weights = [5, 10, -5, -50, -100, 10]
  scores = [closestFood, closestCapsule, closestGhost,
              remainingFood, remainingCapsules, closestScared]

  score = sum(i * j for i, j in zip(scores, weights))

  return score
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
