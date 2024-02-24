# Very naive implementation using Model-Free RL = Q-Learning + Epsilon Greedy Exploration

#States: Old Soft, Old Medium, Old Hard, New Soft, New Medium, New Hard
#Actions: Continue Soft, Continue Medium, Continue Hard, Pit Soft, Pit Medium, Pit Hard
#Rewards: Unknown
#Transitions: Unknown

# Single Car, Entire Race Planning so Finite Horizon Planning - say 50 laps in a race so 50 timesteps

##### Assumptions #####
# -> no tire degradation
# -> single agent setting
# -> no rain
#
# -> no track specific physics
# -> no engine dynamics
# -> considering time and not position, no 'overtaking'
# -> Fully Observable Setting

##### Findings #####


import numpy as np

class QLearning:
    #alpha
    def __init__(self, alpha = 0.1, gamma = 0.99, epsilon = 0.1):
        self.states = ['OS', 'OM', 'OH', 'NS', 'NM', 'NH']  # Old/New Soft, Medium, Hard
        self.actions = ['CS', 'CM', 'CH', 'PS', 'PM', 'PH']
        self.q_table = np.zeros((len(self.states), len(self.actions)))
        self.alpha = alpha # Learning Rate
        self.gamma = gamma # Discount
        self.epsilon = epsilon # Exploration Rate
    
    def choose_action(self, state):
        #Epsilon Greedy Exploration
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)  # Explore
        else:
            state_index = self.states.index(state)
            action_index = np.argmax(self.q_table[state_index])  # Exploit
            action = self.actions[action_index]
        return action
    
    #Q(s, a) = Q(s, a) + alpha[r + gamma * max Q(s', a') - Q(s, a)]
    def update_q_table(self, state, action, reward, next_state):
        state_index = self.states.index(state)
        action_index = self.actions.index(action)
        next_state_index = self.states.index(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index]) #choosing best action from next state
        td_target = reward + self.gamma * self.q_table[next_state_index][best_next_action]
        td_diff = td_target - self.q_table[state_index][action_index]
        self.q_table[state_index][action_index] += self.alpha * td_diff
    
def simulate_race(agent, laps=50):
    state = 'NS'  # Starting with New Soft tires
    total_reward = 0
    tire_age = 0
    pit_stops = 0
    pit_laps = {}
    max_pit_stops_allowed = 2

    for lap in range(1, laps + 1):
        action = agent.choose_action(state)

        if action[-1] == state[-1]:
            reward = -20
        else:
            if 'P' in action:
                pit_stops += 1
                old_tire_type = state[-1]
                new_tire_type = action[-1]

                if pit_stops > max_pit_stops_allowed:
                    reward = -50 
                else:
                    pit_laps[lap] = [old_tire_type, new_tire_type]
                    state = 'N' + new_tire_type
                    tire_age = 0
                    reward = 5
            else: 
                tire_age += 1
                if tire_age > 1 and 'N' in state:  # Tires become old after 1 lap
                    state = 'O' + state[1:]

        if 'S' in state:
            reward += 10 - tire_age  # Faster but wears out quicker
        elif 'M' in state:
            reward += 8 - (tire_age * 0.75)  # Balance between speed and durability
        elif 'H' in state:
            reward += 6 - (tire_age * 0.5)  # Durable but slower

        next_state = state
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    return total_reward, pit_stops, pit_laps


agent = QLearning(alpha = 0.1, gamma = 0.95, epsilon = 0.1)
total_reward, pit_stops, pit_laps = simulate_race(agent)

print(f"Total Reward: {total_reward}")
print(f"Total Pit Stops: {pit_stops}")
print(f"Pit Stops at Lap Numbers with Switch to Tyres: {pit_laps}")

print("Q-Table after the race:")
print(agent.q_table)


