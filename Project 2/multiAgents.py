# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
		newNumFood = successorGameState.getNumFood()
		newGhostStates = successorGameState.getGhostStates()
		newGhostPositions = successorGameState.getGhostPositions()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		oldNumFood = currentGameState.getNumFood()

		score = 0.0

		# If at risk of dying to ghost, severely reduce score
		# Must avoid death at all costs
		for ghostPos in newGhostPositions:
			ghostDist = abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1])
			if ghostDist <= 1.0:
				score = -200.0

		# If pacman can eat food, then just eat that food
		if newNumFood < oldNumFood:
			score += 100.0
			return score

		# If pacman does not need to avoid death or has no food edible next turn
		# Then evaluate where is the nearest food, and set score as the reciprocal to that food
		# So that pacman will want to move towards the nearest food
		nearest = -1
		for x in range(newFood.width):
			for y in range(newFood.height):
				if newFood[x][y] == 1:
					tempDist = abs(x - newPos[0]) + abs(y - newPos[1])
					if nearest == -1 or tempDist < nearest:
						nearest = tempDist

		score += 1.0/nearest
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
	def getValue(self, gameState, currDepth, agent):
		# Once agent is past number, we know 1 ply is over, so reset agent and increment depth
		if agent >= gameState.getNumAgents():
			agent = 0
			currDepth += 1

		# If leaf node or reached desired depth of search
		if gameState.isWin() or gameState.isLose() or currDepth > self.depth:
			return self.evaluationFunction(gameState), "Stop"
		# If pacman turn get max of child states
		elif agent == 0:
			return self.maxValue(gameState, currDepth, agent)
		# Not pacman then ghosts turn so get min of child states
		else:
			return self.minValue(gameState, currDepth, agent)

	def maxValue(self, gameState, currDepth, agent):
		# Variables to hold current "best"
		currMax = float('-inf')
		currAction = "Stop"

		# Check all possible actions for this agent to take
		actions = gameState.getLegalActions(agent)
		for action in actions:
			# Retrieve the value for successor if this action is taken
			successor = gameState.generateSuccessor(agent, action)
			succVal, succAction= self.getValue(successor, currDepth, agent + 1)

			# If better than current "best" update accordingly
			if succVal > currMax:
				currAction = action
				currMax = succVal

		return currMax, currAction

	def minValue(self, gameState, currDepth, agent):
		# Variables to hold current "best"
		currMin = float('inf')
		currAction = "Stop"

		# Check all possible actions for this agent to take
		actions = gameState.getLegalActions(agent)
		for action in actions:
			# Retrieve the value for successor if this action is taken
			successor = gameState.generateSuccessor(agent, action)
			succVal, succAction= self.getValue(successor, currDepth, agent + 1)

			# If better than current "best" update accordingly			
			if succVal < currMin:
				currAction = action
				currMin = succVal

		return currMin, currAction

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
		"""
		value, action = self.getValue(gameState, 1, 0)

		return action

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""
	def getValue(self, gameState, currDepth, bestAlpha, bestBeta, agent):
		# Once agent is past number, we know 1 ply is over, so reset agent and increment depth
		if agent >= gameState.getNumAgents():
			agent = 0
			currDepth += 1

		# If leaf node or reached desired depth of search
		if gameState.isWin() or gameState.isLose() or currDepth > self.depth:
			return self.evaluationFunction(gameState), "Stop"
		# If pacman turn get max of child states
		elif agent == 0:
			return self.maxValue(gameState, currDepth, bestAlpha, bestBeta, agent)
		# Not pacman then ghosts turn so get min of child states
		else:
			return self.minValue(gameState, currDepth, bestAlpha, bestBeta, agent)

	def maxValue(self, gameState, currDepth, bestAlpha, bestBeta, agent):
		# Variables to hold current "best"
		currMax = float('-inf') 
		currAlpha = bestAlpha
		currBeta = bestBeta
		currAction = "Stop"

		# Check all possible actions for this agent to take
		actions = gameState.getLegalActions(agent)
		for action in actions:
			# Retrieve the value for successor if this action is taken
			successor = gameState.generateSuccessor(agent, action)
			succVal, succAction= self.getValue(successor, currDepth, currAlpha, currBeta, agent + 1)

			# If better than current max update accordingly
			if succVal > currMax:
				currAction = action
				currMax = succVal

			# If better than current "best" update accordingly
			if succVal > currAlpha:
				currAlpha = succVal

			# Pruning when alpha > beta
			if currAlpha > currBeta:
				return currAlpha, currAction

		return currMax, currAction

	def minValue(self, gameState, currDepth, bestAlpha, bestBeta, agent):
		# Variables to hold current "best"
		currMin = float('inf') 
		currAlpha = bestAlpha
		currBeta = bestBeta
		currAction = "Stop"	

		# Check all possible actions for this agent to take
		actions = gameState.getLegalActions(agent)
		for action in actions:
			# Retrieve the value for successor if this action is taken
			successor = gameState.generateSuccessor(agent, action)
			succVal, succAction= self.getValue(successor, currDepth, currAlpha, currBeta, agent + 1)

			# If better than current min update accordingly
			if succVal < currMin:
				currAction = action
				currMin = succVal

			# If better than current beta update accordingly			
			if succVal < currBeta:
				currBeta = succVal

			# Pruning when alpha > beta
			if currAlpha > currBeta:
				return currBeta, currAction

		return currMin, currAction

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		value, action = self.getValue(gameState, 1, float('-inf'), float('inf'), 0)

		return action

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""
	def getValue(self, gameState, currDepth, agent):
		# Once agent is past number, we know 1 ply is over, so reset agent and increment depth
		if agent >= gameState.getNumAgents():
			agent = 0
			currDepth += 1

		# If leaf node or reached desired depth of search
		if gameState.isWin() or gameState.isLose() or currDepth > self.depth:
			return self.evaluationFunction(gameState), "Stop"
		# If pacman turn get max of child states
		elif agent == 0:
			return self.maxValue(gameState, currDepth, agent)
		# Not pacman then ghosts turn so get min of child states
		else:
			return self.expectValue(gameState, currDepth, agent)

	def maxValue(self, gameState, currDepth, agent):
		# Variables to hold current "best"
		currMax = float('-inf')
		currAction = "Stop"

		# Check all possible actions for this agent to take
		actions = gameState.getLegalActions(agent)
		for action in actions:
			# Retrieve the value for successor if this action is taken
			successor = gameState.generateSuccessor(agent, action)
			succVal, succAction= self.getValue(successor, currDepth, agent + 1)

			# If better than current max update accordingly
			if succVal > currMax:
				currAction = action
				currMax = succVal

		return currMax, currAction

	def expectValue(self, gameState, currDepth, agent):
		# Variables to hold current "best"
		totalValue = 0
		currAction = "Stop"	

		# Check all possible actions for this agent to take
		actions = gameState.getLegalActions(agent)
		for action in actions:
			# Retrieve the value for successor if this action is taken
			successor = gameState.generateSuccessor(agent, action)
			succVal, succAction= self.getValue(successor, currDepth, agent + 1)

			# Add value to the total
			totalValue += succVal

		# Return the average of all values
		return totalValue / len(actions), currAction

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		value, action = self.getValue(gameState, 1, 0)
		return action

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	"""
	pos = currentGameState.getPacmanPosition()
	foods = currentGameState.getFood().asList()
	numFood = currentGameState.getNumFood()
	numCapsules = len(currentGameState.getCapsules())
	numGhosts = currentGameState.getNumAgents()-1
	ghostStates = currentGameState.getGhostStates()
	ghostPositions = currentGameState.getGhostPositions()
	scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

	score = 0.0

	# If at risk of dying to ghost, severely reduce score
	# Must avoid death at all costs
	if currentGameState.isLose():
		return float('-inf')
	elif currentGameState.isWin():
		return float('inf')

	for i in range(numGhosts):
		ghostDist = abs(ghostPositions[i][0] - pos[0]) + abs(ghostPositions[i][1] - pos[1])
		if scaredTimes[i] > 0:
			score -= 10*ghostDist

	# If pacman does not need to avoid death or has no food edible next turn
	# Then evaluate where is the nearest food, and set score as the reciprocal to that food
	# So that pacman will want to move towards the nearest food
	if(numFood >= 1):
		nearest = float('inf')
		for food in foods:
			tempDist = abs(food[0] - pos[0]) + abs(food[1] - pos[1])
			if tempDist < nearest:
				nearest = tempDist
		score -= nearest*10

	score -= numFood*100
	score -= numCapsules*1000

	return score

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

