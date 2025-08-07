
import json
import csv
import random
import os
import datetime

# --- Configuration ---
# Set the curriculum mode to test different hypotheses:
# 'SUDOKU_ONLY' -> Tests H4 (develops a "cautious logician")
# 'AUT_ONLY'    -> Tests H4 (develops an "exploratory creative")
# 'MIXED'       -> Tests H5 (develops an adaptive agent)
CURRICULUM_MODE = 'SUDOKU_ONLY'
NUM_EPOCHS = 200
LEARNING_RATE = 0.1 # Increased learning rate for more noticeable changes
LOG_FILE = 'results3.csv'

class FERE_Agent:
    """
    Represents the FERE-CRS agent.
    Manages its own CRS weights and learns via a reinforcement learning process.
    """
    def __init__(self, w_r=1.0, w_p=1.0, w_i=1.0, w_c=1.0):
        self.weights = {'R': w_r, 'P': w_p, 'I': w_i, 'C': w_c}
        self._normalize_weights()

    def _normalize_weights(self):
        """ Normalizes weights to sum to 4.0 to prevent runaway values. """
        total = sum(self.weights.values())
        if total > 0:
            factor = 4.0 / total
            self.weights = {k: v * factor for k, v in self.weights.items()}

    def update_weights(self, reward, component_scores):
        """
        Updates CRS weights based on the reward from a completed task.
        This is the core of the heuristic meta-learning.
        [CORRECTED LOGIC]
        """
        # Reinforce weights based on the actual components of the rewarded behavior
        self.weights['R'] += LEARNING_RATE * reward * component_scores.get('R', 0)
        self.weights['P'] += LEARNING_RATE * reward * component_scores.get('P', 0)
        self.weights['I'] += LEARNING_RATE * reward * component_scores.get('I', 0)
        # We penalize cost, so the agent learns to favor efficiency (i.e., low C_cost)
        self.weights['C'] += LEARNING_RATE * reward * (1.0 - component_scores.get('C', 0))

        self.weights = {k: max(0.01, v) for k, v in self.weights.items()}
        self._normalize_weights()

    def _calculate_crs(self, components):
        """ Calculates the CRS for a potential action based on current weights. """
        w = self.weights
        return w['R']*components['R'] + w['P']*components['P'] + w['I']*components['I'] + w['C']*(1.0-components['C'])

    def solve_sudoku_task(self, puzzle):
        """
        Simulates the agent solving a Sudoku puzzle.
        The agent's success and efficiency depend on its current CRS weights.
        """
        print(f"  Attempting Sudoku ({puzzle['difficulty']})...")
        
        # Simulate two candidate strategies: a "safe" move and a "curious" one
        safe_move_components = {'R': 0.95, 'P': 0.8, 'I': 0.1, 'C': 0.2} # High R and P
        curious_move_components = {'R': 0.7, 'P': 0.2, 'I': 0.9, 'C': 0.6} # High I and C

        safe_crs = self._calculate_crs(safe_move_components)
        curious_crs = self._calculate_crs(curious_move_components)

        # Agent chooses the strategy with the higher CRS score
        if safe_crs >= curious_crs:
            chosen_components = safe_move_components
            steps = 15 # Efficient
            print("  Strategy: Safe/Logical move chosen.")
        else:
            chosen_components = curious_move_components
            steps = 25 # Less efficient
            print("  Strategy: Curious/Exploratory move chosen.")

        # Reward is higher for choosing the efficient (safe) strategy
        reward = 1.0 if steps == 15 else 0.2
        print(f"  Sudoku solved. Reward: {reward:.2f}")
        return reward, chosen_components

    def solve_aut_task(self, task):
        """
        Simulates the agent performing the Alternative Uses Test.
        Success depends on generating novel ideas (driven by the I-weight).
        """
        print(f"  Attempting AUT for '{task['object_name']}'...")

        # Simulate two candidate ideas: a "conventional" idea and a "novel" one
        conventional_idea_components = {'R': 0.8, 'P': 0.4, 'I': 0.2, 'C': 0.2} # Low I
        novel_idea_components = {'R': 0.6, 'P': 0.5, 'I': 0.95, 'C': 0.5} # High I

        conventional_crs = self._calculate_crs(conventional_idea_components)
        novel_crs = self._calculate_crs(novel_idea_components)

        # Agent chooses the idea with the higher CRS score
        if novel_crs >= conventional_crs:
            chosen_components = novel_idea_components
            creativity_score = 80 # High reward
            print("  Strategy: Novel idea generated.")
        else:
            chosen_components = conventional_idea_components
            creativity_score = 30 # Low reward
            print("  Strategy: Conventional idea generated.")
        
        reward = creativity_score / 100.0
        print(f"  AUT completed. Creativity Score: {creativity_score}/100. Reward: {reward:.2f}")
        return reward, chosen_components

def main():
    """ Main experiment execution script. """
    print("--- FERE-CRS Phase II: Heuristic Meta-Learning ---")
    
    # 1. Load all curriculum assets
    print("Loading curriculum assets...")
    try:
        with open('sudoku_puzzles.json', 'r') as f:
            sudoku_puzzles = json.load(f)
        with open('creative_tasks.json', 'r') as f:
            aut_tasks = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all .json files are in the same directory as the script.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a file: {e}. Please ensure the JSON files are formatted correctly.")
        return

    # 2. Initialize the agent and log file
    agent = FERE_Agent()
    
    # Setup CSV logging
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'task_type', 'reward', 'w_R', 'w_P', 'w_I', 'w_C'])

    print(f"Starting meta-learning loop for {NUM_EPOCHS} epochs...")
    print(f"Mode: {CURRICULUM_MODE}, Learning Rate: {LEARNING_RATE}\n")
    
    # 3. The Heuristic Meta-Learning Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        task_type, task = None, None
        
        if CURRICULUM_MODE == 'SUDOKU_ONLY':
            task_type, task = 'Sudoku', random.choice(sudoku_puzzles)
        elif CURRICULUM_MODE == 'AUT_ONLY':
            task_type, task = 'AUT', random.choice(aut_tasks)
        else: # MIXED
            if random.random() < 0.5:
                task_type, task = 'Sudoku', random.choice(sudoku_puzzles)
            else:
                task_type, task = 'AUT', random.choice(aut_tasks)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Task: {task_type}")

        if task_type == 'Sudoku':
            reward, components = agent.solve_sudoku_task(task)
        else:
            reward, components = agent.solve_aut_task(task)
            
        agent.update_weights(reward, components)
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            w = agent.weights
            writer.writerow([epoch, task_type, f"{reward:.4f}", f"{w['R']:.4f}", f"{w['P']:.4f}", f"{w['I']:.4f}", f"{w['C']:.4f}"])
            
        print(f"  Updated Weights -> R:{w['R']:.2f}, P:{w['P']:.2f}, I:{w['I']:.2f}, C:{w['C']:.2f}\n")

    print("--- Meta-learning complete ---")
    print(f"Final agent weights: R={agent.weights['R']:.2f}, P={agent.weights['P']:.2f}, I={agent.weights['I']:.2f}, C={agent.weights['C']:.2f}")
    print(f"Results saved to '{LOG_FILE}'. This data can be used to plot the evolution of the agent's cognitive stance.")

if __name__ == '__main__':
    main()