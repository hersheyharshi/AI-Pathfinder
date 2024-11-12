import numpy as np
import random
import matplotlib.pyplot as plt

# Create a 5x5 custom grid world
world_size = 5
harshita_world = np.zeros((world_size, world_size))  # Initialize the grid with zeros
harshita_world[2, 2] = -1  # Set an obstacle at (2, 2)
goal_point = (4, 4)  # Goal at the bottom-right corner

# Initialize Harshita's Q-table with zeros (states x actions)
zeus_q_table = np.zeros((world_size, world_size, 4))  # 4 possible actions: up, down, left, right

# Action mapping: 0 = up, 1 = down, 2 = left, 3 = right
directions = {
    0: (-1, 0),  # Move up
    1: (1, 0),   # Move down
    2: (0, -1),  # Move left
    3: (0, 1)    # Move right
}

# Parameters for Q-learning
alpha_h = 0.1  # Learning rate
gamma_h = 0.9  # Discount factor
epsilon_h = 0.2  # Exploration rate for randomness

# Number of training rounds
training_episodes = 1000

def get_new_position(current_pos, move):
    """Compute the next position based on the current location and action."""
    step = directions[move]
    new_pos = (current_pos[0] + step[0], current_pos[1] + step[1])
    # Ensure the new position is valid and not an obstacle
    if 0 <= new_pos[0] < world_size and 0 <= new_pos[1] < world_size and harshita_world[new_pos[0], new_pos[1]] != -1:
        return new_pos
    return current_pos  # Return the same position if the move is invalid

def calculate_reward(pos):
    """Determine the reward for reaching a given position."""
    if pos == goal_point:
        return 100  # Reward for reaching the goal
    elif harshita_world[pos[0], pos[1]] == -1:
        return -100  # Penalty for hitting an obstacle
    else:
        return -1  # Penalty for each move to shorten the path

# Run Q-learning training
for ep in range(training_episodes):
    current_pos = (0, 0)  # Start at the top-left corner
    reached_goal = False

    while not reached_goal:
        # Choose action: explore or exploit
        if random.uniform(0, 1) < epsilon_h:
            chosen_action = random.choice(list(directions.keys()))  # Random action
        else:
            chosen_action = np.argmax(zeus_q_table[current_pos[0], current_pos[1]])  # Best action based on Q-table

        # Take the action and observe the outcome
        next_pos = get_new_position(current_pos, chosen_action)
        reward = calculate_reward(next_pos)

        # Update the Q-value using the Q-learning formula
        zeus_q_table[current_pos[0], current_pos[1], chosen_action] += alpha_h * (
            reward + gamma_h * np.max(zeus_q_table[next_pos[0], next_pos[1]]) - zeus_q_table[current_pos[0], current_pos[1], chosen_action]
        )

        # Move to the next state
        current_pos = next_pos

        # Check if the goal is reached
        if current_pos == goal_point:
            reached_goal = True

# Display the learned Q-table
print("Zeus' Q-table after training:")
print(zeus_q_table)

# Extract the optimal route
current_pos = (0, 0)
best_path = [current_pos]
while current_pos != goal_point:
    best_step = np.argmax(zeus_q_table[current_pos[0], current_pos[1]])
    current_pos = get_new_position(current_pos, best_step)
    best_path.append(current_pos)

print("Harshita's optimal path from start to goal:", best_path)

# Function to visualize the path
def show_path_on_grid(path):
    path_visual = harshita_world.copy()
    for step in path:
        path_visual[step[0], step[1]] = 2  # Mark the path
    path_visual[0, 0] = 3  # Mark start point
    path_visual[goal_point[0], goal_point[1]] = 4  # Mark goal point

    plt.imshow(path_visual, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Harshita's Optimal Path Visualization")
    plt.show()

# Display the path
show_path_on_grid(best_path)