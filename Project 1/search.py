# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
	"""
	This class outlines the structure of a search problem, but doesn't implement
	any of the methods (in object-oriented terminology: an abstract class).

	You do not need to change anything in this class, ever.
	"""

	def getStartState(self):
		"""
		Returns the start state for the search problem
		"""
		util.raiseNotDefined()

	def isGoalState(self, state):
		"""
		  state: Search state

		Returns True if and only if the state is a valid goal state
		"""
		util.raiseNotDefined()

	def getSuccessors(self, state):
		"""
		  state: Search state

		For a given state, this should return a list of triples,
		(successor, action, stepCost), where 'successor' is a
		successor to the current state, 'action' is the action
		required to get there, and 'stepCost' is the incremental
		cost of expanding to that successor
		"""
		util.raiseNotDefined()

	def getCostOfActions(self, actions):
		"""
		 actions: A list of actions to take

		This method returns the total cost of a particular sequence of actions.  The sequence must
		be composed of legal moves
		"""
		util.raiseNotDefined()


def tinyMazeSearch(problem):
	"""
	Returns a sequence of moves that solves tinyMaze.  For any other
	maze, the sequence of moves will be incorrect, so only use this for tinyMaze
	"""
	from game import Directions
	s = Directions.SOUTH
	w = Directions.WEST
	return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
	"""
	Search the deepest nStates in the search tree first

	Your search algorithm needs to return a list of actions that reaches
	the goal.  Make sure to implement a graph search algorithm

	To get started, you might want to try some of these simple commands to
	understand the search problem that is being passed in:
	"""

	# Initial Conditions for DFS pathfinding
	# Fringe is a stack
	fringe = util.Stack()
	fringe.push(((problem.getStartState(), None, 0), None))
	path = []
	visited = []

	# Continue to loop until path is found or not found
	while(1):
		# If fringe is empty, no path is found, return None
		if fringe.isEmpty():
			return None

		# Get next nState from fringe to evaluate	
		# nState = s, prev
		nState = fringe.pop()

		# If we found the goal, make the path and return
		if problem.isGoalState(nState[0][0]) == 1:
			# While there is a previous nState, keep inserting its direction
			while(nState[1]):
				path.insert(0, nState[0][1])
				nState = nState[1]
			return path

		# If it's not the goal and we haven't visited before, process it
		elif not nState[0][0] in visited:
			successors = problem.getSuccessors(nState[0][0])
			# Add current nState to visited list
			visited.append(nState[0][0])
			# Add all successors to the fringe
			for succ in successors:
				fringe.push((succ, nState))

def breadthFirstSearch(problem):
	"""
	Search the shallowest nStates in the search tree first.
	"""

	# Initial Conditions for BFS pathfinding
	# Fringe is a queue
	fringe = util.Queue()
	fringe.push(((problem.getStartState(), None, 0), None))
	path = []
	visited = []

	# Continue to loop until path is found or not found
	while(1):
		# If fringe is empty, no path is found, return None
		if fringe.isEmpty():
			return None

		# Get next nState from fringe to evaluate	
		# nState = s, prev
		nState = fringe.pop()

		# If we found the goal, make the path and return
		if problem.isGoalState(nState[0][0]) == 1:
			# While there is a previous nState, keep inserting its direction
			while(nState[1]):
				path.insert(0, nState[0][1])
				nState = nState[1]
			return path

		# If it's not the goal and we haven't visited before, process it
		elif not nState[0][0] in visited:
			successors = problem.getSuccessors(nState[0][0])
			# Add current nState to visited list
			visited.append(nState[0][0])
			# Add all successors to the fringe
			for succ in successors:
				fringe.push((succ, nState))
				
def uniformCostSearch(problem):
	"""
	Search the nState of least total cost first. 
	"""

	# Initial Conditions for UCS pathfinding
	# Fringe is a priority queue
	fringe = util.PriorityQueue()
	fringe.push(((problem.getStartState(), None, 0), None, 0), 0)
	path = []
	visited = []

	# Continue to loop until path is found or not found
	while(1):
		# If fringe is empty, no path is found, return None
		if fringe.isEmpty():
			return None

		# Get next nState from fringe to evaluate	
		# nState = s, prev
		nState = fringe.pop()

		# If we found the goal, make the path and return
		if problem.isGoalState(nState[0][0]) == 1:
			# While there is a previous nState, keep inserting its direction
			while(nState[1]):
				path.insert(0, nState[0][1])
				nState = nState[1]
			return path

		# If it's not the goal and we haven't visited before, process it
		elif not nState[0][0] in visited:
			successors = problem.getSuccessors(nState[0][0])
			# Add current nState to visited list
			visited.append(nState[0][0])
			# Add all successors to the fringe with total values
			for succ in successors:
				fringe.push((succ, nState, nState[2] + succ[2]), nState[2] + succ[2])

def nullHeuristic(state, problem=None):
	"""
	A heuristic function estimates the cost from the current state to the nearest
	goal in the provided SearchProblem.  This heuristic is trivial.
	"""
	return 0

def aStarSearch(problem, heuristic=nullHeuristic):
	"""
	Search the nState that has the lowest combined cost and heuristic first.
	"""
 
	# Fringe is a priority queue
	fringe = util.PriorityQueue()
	fringe.push(((problem.getStartState(), None, 0), None, 0), 0)
	path = []
	visited = []

	# Continue to loop until path is found or not found
	while(1):
		# If fringe is empty, no path is found, return None
		if fringe.isEmpty():
			return None

		# Get next nState from fringe to evaluate	
		# nState = s, prev
		nState = fringe.pop()

		# If we found the goal, make the path and return
		if problem.isGoalState(nState[0][0]) == 1:
			# While there is a previous nState, keep inserting its direction
			while(nState[1]):
				path.insert(0, nState[0][1])
				nState = nState[1]
			return path

		# If it's not the goal and we haven't visited before, process it
		elif not nState[0][0] in visited:
			successors = problem.getSuccessors(nState[0][0])
			# Add current nState to visited list
			visited.append(nState[0][0])
			# Add all successors to the fringe with total values
			for succ in successors:
				fringe.push((succ, nState, nState[2] + succ[2]), nState[2] + succ[2] + heuristic(succ[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
