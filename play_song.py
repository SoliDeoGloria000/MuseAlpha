# play_song.py
import numpy as np
import os
from environment import PianoEnvironment
from agent import ImprovedDQNAgent  # <--- CHANGE 1: Use the correct class name
import pickle

def action_to_string(action):
    """Converts a parsed action dictionary back into a string representation."""
    if action['type'] == 'note':
        # In environment_v2, single notes are still in a list
        return action.get('note', '')
    elif action['type'] == 'chord':
        return f"[{''.join(action.get('notes', []))}]"
    elif action['type'] == 'pause':
        return '|' * action.get('duration', 1)
    return ''

if __name__ == "__main__":
    song_to_play = "4|[ute]|s|[o1]|[wur]||4|[ute]||1|[wur]||4|[utfe]|h|[g1]"

    print("--- Loading Environment and Agent ---")
    env = PianoEnvironment(song_to_play)
    # The state size must match the one used for training
    state_size = env.state_space_size + 12
    action_size = env.action_space_size

    # Use the correct agent class
    agent = ImprovedDQNAgent(state_size, action_size)

    # <--- CHANGE 2: Use the checkpoint file from the multiprocessing trainer
    checkpoint_filename = "piano_agent_multiprocess.pkl"

    if not os.path.exists(checkpoint_filename):
        print(f"Could not find checkpoint file: {checkpoint_filename}")
        print("Please train the agent first by running train_multiprocess.py")
    else:
        print(f"Loading checkpoint from {checkpoint_filename}...")
        with open(checkpoint_filename, 'rb') as f:
            checkpoint_data = pickle.load(f)
            agent.model.set_weights(checkpoint_data['model_weights'])
            agent.target_model.set_weights(checkpoint_data['target_model_weights'])
            agent.epsilon = 0.0 # Set to exploitation mode

        print("\n--- Agent Loaded. Generating Performance ---")

        state = env.reset()
        done = False
        agent_actions = []

        max_steps = env.state_space_size * 2
        steps_taken = 0

        while not done and steps_taken < max_steps:
            # Apply the same action masking that was used during training so the
            # agent only considers the valid action for the current timestep.
            mask = env.get_action_mask()
            action_id = agent.act(state, mask)

            action_details = env.actions[action_id]
            agent_actions.append(action_details)

            next_state, _, done = env.step(action_id)
            state = next_state
            steps_taken += 1

        print(f"Performance generated with {len(agent_actions)} actions.")
        performance_string = " ".join([action_to_string(a) for a in agent_actions])

        output_filename = "agent_performance.txt"
        with open(output_filename, "w") as f:
            f.write(performance_string)

        print(f"\n--- Performance saved to {output_filename} ---")
        print("\n--- Performance Preview ---")
        print(performance_string[:200] + "...")