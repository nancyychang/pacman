# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
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
        for ghost in newGhostStates:
          if ghost.getPosition() == newPos:
            if ghost.scaredTimer == 0:
              return -float('inf')

        if action == 'Stop':
          return -float('inf')

        manhattanDistances = []

        foodCoords = currentGameState.getFood().asList()
        for food in foodCoords:
          foodX = -abs(food[0] - newPos[0])
          foodY = -abs(food[1] - newPos[1])
          manhattanDistances.append(foodX + foodY)

        return max(manhattanDistances)

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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxScore, optimalMove = self.maxValue(gameState, self.depth)

        return optimalMove

    def maxValue(self, state, depth):
        if (state.isWin() or state.isLose() or depth == 0):
          return self.evaluationFunction(state), "Stop"

        legalMoves = state.getLegalActions()
        scoresList = [self.minValue(state.generateSuccessor(self.index, move), 1, depth) for move in legalMoves]
        maxScore = max(scoresList)

        for index in range(len(scoresList)):
          if scoresList[index] == maxScore:
            chosenIndex = index
            break

        return maxScore, legalMoves[chosenIndex]

    def minValue(self, state, index, depth):  
        if (state.isWin() or state.isLose() or depth == 0):
          return self.evaluationFunction(state), "Stop"

        legalMoves = state.getLegalActions(index)

        if index != state.getNumAgents() - 1:
          scoresList =[self.minValue(state.generateSuccessor(index, move), index+1, depth) for move in legalMoves]
        else:
          scoresList =[self.maxValue(state.generateSuccessor(index, move), depth-1) for move in legalMoves]
        
        minScore = min(scoresList)

        for index in range(len(scoresList)):
          if scoresList[index] == minScore:
            chosenIndex = index
            break

        return minScore, legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxBest = -float("inf")
        minBest = float("inf")
        return self.maxValue(gameState, 0, 1, maxBest, minBest)

    def maxValue(self, state, index, depth, maxBest, minBest):
        if state.isWin() or state.isLose() or depth == 0:
          return self.evaluationFunction(state)

        legalMoves = state.getLegalActions()

        maxScore = -float("inf")
        for move in legalMoves:
          tempMaxScore = self.minValue(state.generateSuccessor(self.index, move), 1, depth, maxBest, minBest)
          if tempMaxScore > minBest:
            return tempMaxScore
          if tempMaxScore > maxScore:
            maxScore = tempMaxScore
            action = move
          maxBest = max(maxBest, maxScore)

        if depth == 1:
          return action
        else:
          return maxScore

    def minValue(self, state, index, depth, maxBest, minBest):  
        
        if state.isWin() or state.isLose() or depth == 0:
          return self.evaluationFunction(state)

        legalMoves = state.getLegalActions(index)

        minScore = float("inf")
        for move in legalMoves:
          if index != state.getNumAgents() - 1:
            tempMinScore = self.minValue(state.generateSuccessor(index, move), index+1, depth, maxBest, minBest)
          else:
            if depth == self.depth:
              tempMinScore = self.evaluationFunction(state.generateSuccessor(index, move))
            else:
              tempMinScore = self.maxValue(state.generateSuccessor(index, move), 0, depth+1, maxBest, minBest)
          if tempMinScore < maxBest:
            return tempMinScore
          if tempMinScore < minScore:
            minScore = tempMinScore
          minBest = min(minBest, minScore)
        return minScore

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
        return self.maxValue(gameState, 0, 1)

    def maxValue(self, state, index, depth):
      if state.isWin() or state.isLose() or depth == 0:
        return self.evaluationFunction(state)

      legalMoves = state.getLegalActions()

      maxScore = -float("inf")
      for move in legalMoves:
        tempMaxScore = self.expectedValue(state.generateSuccessor(0, move), 1, depth)
        if tempMaxScore > maxScore:
          maxScore = tempMaxScore
          action = move

      if depth == 1:
        return action
      else:
        return maxScore

    def expectedValue(self, state, index, depth):
      if state.isWin() or state.isLose() or depth == 0:
          return self.evaluationFunction(state)

      legalMoves = state.getLegalActions(index)

      expectedVal = 0
      for move in legalMoves:
        if index == state.getNumAgents() - 1:
          if depth == self.depth:
            tempExpScore = self.evaluationFunction(state.generateSuccessor(index, move))
          else:
            tempExpScore = self.maxValue(state.generateSuccessor(index, move), 0, depth+1)
        else:
          tempExpScore = self.expectedValue(state.generateSuccessor(index, move), index+1, depth)
      
        expectedVal += tempExpScore
      return expectedVal/len(legalMoves)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      Found the Manhattan Distances from Pacman to each food pellet
      and the Manhattan Distances from Pacman to each ghost (took reciprocal of distance to ghosts)

      Evaluation function returns the sum of the maximum distance of Pacman from food, minimum distance
      of Pacman from ghost, and the numerical score of the current game state and substracting the number of ghosts 
      minus the number of ghosts that are scared multiplied by an arbitrary number 50. 

    """
    "*** YOUR CODE HERE ***"
    foodDist = []
    ghostDist = []
    score = 0
    foodList = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    pacman = currentGameState.getPacmanPosition()
    numGhostsScared = 0

    for food in foodList:
      x, y = abs(food[0] - pacman[0]), abs(food[1] - pacman[1])
      foodDist.append(-(x + y))

    if not foodDist: # no food
      foodDist.append(0) 

    for ghost in ghosts:
      if ghost.scaredTimer == 0:
        numGhostsScared += 1
        ghostDist.append(0)
        continue # don't care about ghosts if they're scared

      ghostPos = ghost.getPosition()
      x, y = abs(ghostPos[0] - pacman[0]), abs(ghostPos[1] - pacman[1])
      if x + y > 0:
        ghostDist.append(-1.0/(x+y)) # reciprocal
      else:
        ghostDist.append(0)
    return max(foodDist) + min(ghostDist) + currentGameState.getScore() - 50*(len(ghosts) - numGhostsScared)

# Abbreviation
better = betterEvaluationFunction

