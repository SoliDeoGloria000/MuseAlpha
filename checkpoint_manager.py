import pickle
import os
from tensorflow.keras.models import load_model

def save_checkpoint(agent, filename="piano_agent_checkpoint.pkl"):
    """
    Saves the agent's state, including model weights and epsilon.
    
    Args:
        agent: The DQNAgent object to save.
        filename (str): The file to save the checkpoint to.
    """
    checkpoint_data = {
        'model_weights': agent.model.get_weights(),
        'epsilon': agent.epsilon
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"\nCheckpoint saved to {filename}")

def load_checkpoint(agent, filename="piano_agent_checkpoint.pkl"):
    """
    Loads the agent's state from a checkpoint file.

    Args:
        agent: The DQNAgent object to load the state into.
        filename (str): The file to load the checkpoint from.
    
    Returns:
        bool: True if a checkpoint was successfully loaded, False otherwise.
    """
    if os.path.exists(filename):
        print(f"--- Found existing checkpoint: {filename} ---")
        try:
            with open(filename, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            agent.model.set_weights(checkpoint_data['model_weights'])
            agent.epsilon = checkpoint_data['epsilon']
            print(f"Checkpoint successfully loaded. Resuming with epsilon = {agent.epsilon:.2f}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            return False
    else:
        print("--- No checkpoint found. Starting fresh training. ---")
        return False
