# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
	"""
		* Please read learningAgents.py before reading this.*

		A ValueIterationAgent takes a Markov decision process
		(see mdp.py) on initialization and runs value iteration
		for a given number of iterations using the supplied
		discount factor.
	"""
	def __init__(self, mdp, discount = 0.9, iterations = 100):
		"""
		  Your value iteration agent should take an mdp on
		  construction, run the indicated number of iterations
		  and then act according to the resulting policy.

		  Some useful mdp methods you will use:
			  mdp.getStates()
			  mdp.getPossibleActions(state)
			  mdp.getTransitionStatesAndProbs(state, action)
			  mdp.getReward(state, action, nextState)
			  mdp.isTerminal(state)
		"""
		self.mdp = mdp
		self.discount = discount
		self.iterations = iterations
		self.values = util.Counter() # A Counter is a dict with default 0

		# Continue updating V values for given number of iterations
		for i in range(self.iterations):
			# Update V values for every state
			newVals = util.Counter()
			for state in self.mdp.getStates():
				# Calculate Q values based on all actions
				QVal = util.Counter()

				# If terminal state, set Q to 0, else get max Q
				if self.mdp.isTerminal(state):
					newVals[state] = 0.0
				else:
					for action in self.mdp.getPossibleActions(state):
						QVal[action] = self.getQValue(state, action)
					# Update new V values to be max of calculated Q values
					newVals[state] = max(QVal.values())

			self.values = newVals

	def getValue(self, state):
		"""
		  Return the value of the state (computed in __init__).
		"""
		return self.values[state]


	def computeQValueFromValues(self, state, action):
		"""
		  Compute the Q-value of action in state from the
		  value function stored in self.values.
		"""
		QVal = 0.0

		# Get Transition states and their probabilities
		TVals = self.mdp.getTransitionStatesAndProbs(state, action)

		# Sum all possible transition states
		for TVal in TVals:
			# Get state and probability of entering
			TState = TVal[0]
			TProb = TVal[1]

			# Get the reward value for entering this transition state
			RVal = self.mdp.getReward(state, action, TState)

			# Add the value to total QVal
			QVal += TProb * (RVal + self.discount * self.values[TState])

		return QVal

	def computeActionFromValues(self, state):
		"""
		  The policy is the best action in the given state
		  according to the values currently stored in self.values.

		  You may break ties any way you see fit.  Note that if
		  there are no legal actions, which is the case at the
		  terminal state, you should return None.
		"""
		# If terminal state return None since no legal action
		if self.mdp.isTerminal(state):
			return None

		# Calculate Q values based on all actions
		QVal = util.Counter()

		# Get Q values
		for action in self.mdp.getPossibleActions(state):
			QVal[action] = self.getQValue(state, action)

		# Return argmax Q
		return QVal.argMax()

	def getPolicy(self, state):
		return self.computeActionFromValues(state)

	def getAction(self, state):
		"Returns the policy at the state (no exploration)."
		return self.computeActionFromValues(state)

	def getQValue(self, state, action):
		return self.computeQValueFromValues(state, action)
