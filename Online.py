#####
# States: continuous statespace containing information about the race position and performance of all the drivers
# Actions: discrete action-space, representing the decision to perform a pit-stop or not (along with which type)
# Using online planning


import numpy as np
import math

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.available_actions()

    def available_actions(self):
        return ['no_pit', 'pit_soft', 'pit_medium', 'pit_hard']
    
    def ucb1(self, total_visits):
        if self.visits == 0:
            return float('inf')  # Ensure unvisited nodes are prioritized
        return (self.wins / self.visits) + 2 * math.sqrt(math.log(total_visits) / self.visits)

class MCTS:
    def __init__(self, simulations=1000):
        self.simulations = simulations
    
    def select_node(self, node):
        while node.children:
            visits = sum(child.visits for child in node.children)
            node = max(node.children, key=lambda x: x.ucb1(visits))
        return node
    
    def expand(self, node):
        action = node.untried_actions.pop()
        child_state = self.transition(node.state, action) 
        child_node = Node(state=child_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):
        # Simplified simulation that randomly assigns wins/losses
        return np.random.choice([True, False])

    def backpropagate(self, node, win):
        while node:
            node.visits += 1
            if win:
                node.wins += 1
            node = node.parent
    
    def transition(self, state, action):
        # Simplified transition model for the race scenario
        return state  # Placeholder for actual state transition logic

    def run(self, root_state):
        root_node = Node(state=root_state)
        
        for _ in range(self.simulations):
            node = self.select_node(root_node)
            if node.untried_actions:
                node = self.expand(node)
            win = self.simulate(node)
            self.backpropagate(node, win)
        
        # Return the action of the child with the highest win ratio
        return max(root_node.children, key=lambda c: c.wins / c.visits if c.visits else 0).action

initial_root_state = {"position": 5, "tire_condition": "medium", "laps_remaining": 20}
mcts = MCTS(simulations=100)

# Run MCTS to determine the next action
recommended_action = mcts.run(initial_root_state)
print(recommended_action)
