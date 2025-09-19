# debug_agent.py
import numpy as np
import os
import pickle
from environment import PianoEnvironment
from agent import ImprovedDQNAgent

CHECKPOINT_PATH = 'piano_agent_multiprocess.pkl'
SONG_STRING = "4|[ute]|s|[o1]|[wur]||4|[ute]||1|[wur]||4|[utfe]|h|[g1]"

# --- Main Debugging Logic ---
if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        exit()

    print("--- Loading Environment and Trained Agent ---")
    env = PianoEnvironment(SONG_STRING)
    state_size = env.state_space_size + 12
    action_size = env.action_space_size
    agent = ImprovedDQNAgent(state_size, action_size)

    # Load the weights from your last training run
    with open(CHECKPOINT_PATH, 'rb') as f:
        checkpoint = pickle.load(f)
        agent.model.set_weights(checkpoint['model_weights'])
    print("Agent loaded successfully.")

    # --- Let's investigate the moment of failure ---
    # The song is "4 | [ute] | s...". The agent fails at the 3rd action (index 2).
    failure_step = 4
    env.current_step = failure_step
    
    print(f"\n--- Analyzing Agent's Decision at Step {failure_step+1} ---")
    
    # Get the state at this exact moment
    state = env._get_state()
    
    # Get the Q-values (the agent's predicted score for each action) from the model
    predicted_q_values = agent.model.predict(state[np.newaxis, :], verbose=0)[0]
    
    # Find the action the agent would choose (the one with the highest score)
    chosen_action_id = np.argmax(predicted_q_values)
    
    # Get the details of the correct action for comparison
    correct_action = env.parsed_song[failure_step]
    correct_action_key = env._canonical_key(correct_action)
    correct_action_id = env.action_to_id[correct_action_key]

    print(f"The CORRECT action is: '{env.actions[correct_action_id]}'")
    print(f"The AGENT chose action: '{env.actions[chosen_action_id]}'")
    
    print("\n--- Agent's Predicted Scores (Q-Values) for All Actions ---")
    # Display all actions and their scores, sorted from best to worst
    action_scores = sorted(
        [(q_val, env.actions[i]) for i, q_val in enumerate(predicted_q_values)],
        key=lambda item: item[0],
        reverse=True
    )
    
    for score, action in action_scores:
        is_correct = ">> CORRECT" if action == env.actions[correct_action_id] else ""
        is_chosen = ">> CHOSEN BY AGENT" if action == env.actions[chosen_action_id] else ""
        print(f"Score: {score:8.4f} | Action: {str(action):<20} {is_correct}{is_chosen}")