# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class StateNode:
    def __init__(self, state, backwards_cost = 0, heuristic_value = 0):
        self.state = state
        self.backwards_cost = backwards_cost
        self.heuristic_value = heuristic_value
    
    def total_cost(self):
        return self.backwards_cost + self.heuristic_value

class FringeNode:
    def __init__(self, prev, state_node: StateNode, action):
        self.prev = prev
        self.state_node = state_node
        self.action = action

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe_stack = util.Stack()
    fringe_stack.push(FringeNode(None,StateNode(problem.getStartState()),None))
    action_list = []
    visited = set()
    while not fringe_stack.isEmpty():
        fringe = fringe_stack.pop()
        state = fringe.state_node.state
        visited.add(state)
        if problem.isGoalState(state):
            currentfringe = fringe
            while currentfringe.action != None:
                action_list.append(currentfringe.action)
                currentfringe = currentfringe.prev
            action_list.reverse()
            return action_list
        successors = problem.getSuccessors(state)
        for successor in successors:
            if successor[0] not in visited:
                fringe_stack.push(FringeNode(fringe,StateNode(successor[0]),successor[1]))
    return action_list

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe_queue = util.Queue()
    fringe_queue.push(FringeNode(None,StateNode(problem.getStartState()),None))
    action_list = []
    visited = set(problem.getStartState())
    while not fringe_queue.isEmpty():
        fringe = fringe_queue.pop()
        state = fringe.state_node.state
        if problem.isGoalState(state):
            currentfringe = fringe
            while currentfringe.action != None:
                action_list.append(currentfringe.action)
                currentfringe = currentfringe.prev
            action_list.reverse()
            return action_list
        successors = problem.getSuccessors(state)
        for successor in successors:
            if successor[0] not in visited:
                visited.add(successor[0])
                fringe_queue.push(FringeNode(fringe,StateNode(successor[0]),successor[1]))
    return action_list

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe_priority_queue = util.PriorityQueueWithFunction(
        lambda fringe_node: fringe_node.state_node.total_cost())
    fringe_priority_queue.push(FringeNode(None,StateNode(problem.getStartState(),0),None))
    list = []
    visited = set()
    while not fringe_priority_queue.isEmpty():
        fringe = fringe_priority_queue.pop()
        state = fringe.state_node.state
        backwards_cost = fringe.state_node.backwards_cost
        if problem.isGoalState(state):
            currentfringe = fringe
            while currentfringe.action != None:
                list.append(currentfringe.action)
                currentfringe = currentfringe.prev
            list.reverse()
            return list
        if state in visited:
            continue
        visited.add(state)
        successors = problem.getSuccessors(state)
        for successor in successors:
            if successor[0] not in visited:
                fringe_priority_queue.push(
                    FringeNode(fringe,
                               StateNode(successor[0], backwards_cost + successor[2]),
                               successor[1]))
    return list

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    fringe_priority_queue = util.PriorityQueueWithFunction(
        lambda fringe_node: fringe_node.state_node.total_cost())
    fringe_priority_queue.push(
        FringeNode(None,
                   StateNode(start_state,0,heuristic(start_state,problem)),
                   None))
    list = []
    visited = set()
    while not fringe_priority_queue.isEmpty():
        fringe = fringe_priority_queue.pop()
        state = fringe.state_node.state
        backwards_cost = fringe.state_node.backwards_cost
        if problem.isGoalState(state):
            currentfringe = fringe
            while currentfringe.action != None:
                list.append(currentfringe.action)
                currentfringe = currentfringe.prev
            list.reverse()
            return list
        if state in visited:
            continue
        visited.add(state)
        successors = problem.getSuccessors(state)
        for successor in successors:
            successor_state, successor_action, successor_stepCost = successor
            if successor_state not in visited:
                fringe_priority_queue.push(
                    FringeNode(fringe,
                               StateNode(successor_state,
                                         backwards_cost + successor_stepCost,
                                         heuristic(successor_state,problem)),
                               successor_action))
    return list


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
