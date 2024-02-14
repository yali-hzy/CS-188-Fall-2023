# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            QValue = dict()
            states = self.mdp.getStates()
            for state in states:
                QValue[state] = self.computeQValueFromValues(state,
                                                             self.computeActionFromValues(state))
            for state in states:
                self.values[state] = QValue[state]


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
        "*** YOUR CODE HERE ***"
        if action == None:
            # print(f"leaves {state} {self.values[state]}")
            return self.values[state]
        successors = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0
        for successor in successors:
            nextState, prob = successor
            QValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        # print(f"QValue {state} {action} {QValue}")
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        maxQValue = -float('inf')
        for action in self.mdp.getPossibleActions(state):
            QValue = self.computeQValueFromValues(state, action)
            if QValue > maxQValue:
                maxQValue = QValue
                bestAction = action
        # print(state, maxQValue, bestAction)
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {state:set() for state in states}
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for successor, _ in self.mdp.getTransitionStatesAndProbs(state,action):
                    predecessors[successor].add(state)
        
        priorityQueue = util.PriorityQueue()
        for s in states:
            if self.mdp.isTerminal(s):
                continue
            maxQValue = max([self.computeQValueFromValues(s, action) 
                             for action in self.mdp.getPossibleActions(s)])
            diff = abs(self.values[s] - maxQValue)
            priorityQueue.push(s, -diff)

        for _ in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            s = priorityQueue.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = self.computeQValueFromValues(s, self.computeActionFromValues(s))
            for p in predecessors[s]:
                maxQValue = max([self.computeQValueFromValues(p, action) 
                                 for action in self.mdp.getPossibleActions(p)])
                diff = abs(self.values[p] - maxQValue)
                if diff > self.theta:
                    priorityQueue.update(p, -diff)

