# train_multiprocess.py

import multiprocessing
import time
import os
from collections import deque
import numpy as np
import pickle # Using pickle for saving checkpoints
from torch.utils.tensorboard import SummaryWriter

# Make sure these files are in the same directory
from environment import PianoEnvironment
from agent import ImprovedDQNAgent # Using your improved agent

CHECKPOINT_PATH = 'piano_agent_multiprocess.pkl'

def warm_start_supervised(agent, env, epochs=2):
    """Pre-trains the agent by teaching it the VALUE of correct vs. wrong actions."""
    X, Y = [], []
    
    # Get the reward values directly from the environment instance
    correct_reward = env.R_CORRECT 
    wrong_reward = env.R_WRONG

    # Create a labeled dataset: state -> target Q-values
    original_pos = env.current_step
    for pos in range(env.state_space_size):
        env.current_step = pos
        state = env._get_state()
        
        # Create a target array where incorrect actions have a negative value
        target_q_values = np.full(env.action_space_size, wrong_reward, dtype=np.float32)
        
        # Get the ID of the single correct action
        correct_action_id = np.argmax(env.get_action_mask())
        
        # Set the value for the correct action to be high
        target_q_values[correct_action_id] = correct_reward
        
        X.append(state)
        Y.append(target_q_values)

    env.current_step = original_pos # Restore env state
    
    X, Y = np.array(X), np.array(Y)
    
    if X.shape[0] > 0:
        print(f"--- Warm-starting with {len(X)} samples... ---")
        agent.model.fit(X, Y, epochs=epochs, batch_size=32, verbose=1)
        agent.update_target_network()

# --- 1. The Worker's Task ---
# This function is executed by each parallel process.
def _unpack_weight_packet(packet):
    """Extract weights and epsilon information sent from the learner."""
    if isinstance(packet, dict):
        weights = packet.get('weights')
        epsilon = packet.get('epsilon')
        return weights, epsilon
    # Backwards compatibility: older packets only contained the weights
    return packet, None


def worker_task(song_string, experience_queue, model_weights_queue, worker_id, episodes_per_worker, max_steps):
    """
    A worker's job is to:
    1. Initialize its own environment and agent.
    2. Get the latest model weights from the central learner.
    3. Play episodes to generate experience.
    4. Send that experience back to the central learner.
    """
    # Prevent workers from trying to use the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env = PianoEnvironment(song_string, max_steps=max_steps)
    agent = ImprovedDQNAgent(env.state_space_size + 12, env.action_space_size)
    
    # Get initial weights from the main process
    try:
        initial_packet = model_weights_queue.get()
        initial_weights, initial_epsilon = _unpack_weight_packet(initial_packet)
        agent.model.set_weights(initial_weights)
        agent.target_model.set_weights(initial_weights)
        if initial_epsilon is not None:
            agent.epsilon = initial_epsilon
    except Exception as e:
        print(f"[Worker {worker_id}] Error getting initial weights: {e}")
        return

    for episode in range(episodes_per_worker):
        # Periodically check for updated weights from the learner
        try:
            new_packet = model_weights_queue.get_nowait()
            new_weights, new_epsilon = _unpack_weight_packet(new_packet)
            agent.model.set_weights(new_weights)
            agent.target_model.set_weights(new_weights)
            if new_epsilon is not None:
                agent.epsilon = new_epsilon
        except:
            pass # No new weights available, just continue

        state = env.reset()
        done = False
        episode_experiences = []
        total_reward = 0

        while not done:
            mask = env.get_action_mask()
            action = agent.act(state, mask)
            next_state, reward, done = env.step(action)
            next_mask = env.get_action_mask()
            episode_experiences.append((state, action, reward, next_state, done, next_mask))
            state = next_state
            total_reward += reward

        # Send the collected experiences and reward back to the main process
        experience_queue.put((episode_experiences, total_reward))

# --- 2. The Main Learner Process ---
if __name__ == "__main__":
    # It's important to freeze support for multiprocessing on some OSes
    multiprocessing.freeze_support()
    
    writer = SummaryWriter('logs/piano_rl_run')

    song_to_learn = "4|[ute]|s|[o1]|[wur]||4|[ute]||1|[wur]||4|[utfe]|h|[g1]" # Using a shorter song for demonstration
    
    print("--- Initializing Central Agent (Learner) ---")
    temp_env = PianoEnvironment(song_to_learn)
    state_size = temp_env.state_space_size + 12
    action_size = temp_env.action_space_size

    central_agent = ImprovedDQNAgent(state_size, action_size)

    # Warm-start the policy with the correct action at each position
    warm_start_supervised(central_agent, temp_env, epochs=2)
    latest_weights = central_agent.model.get_weights()

    # Load checkpoint if it exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        with open(CHECKPOINT_PATH, 'rb') as f:
            checkpoint_data = pickle.load(f)
            central_agent.model.set_weights(checkpoint_data['model_weights'])
            central_agent.target_model.set_weights(checkpoint_data['target_model_weights'])
            central_agent.epsilon = checkpoint_data['epsilon']
            latest_weights = central_agent.model.get_weights()

    # --- Configuration ---
    num_workers = max(1, multiprocessing.cpu_count() - 1) # Leave 2 cores for the learner and OS
    total_episodes = 100000
    episodes_per_worker = total_episodes // num_workers
    update_weights_interval = 10 # How often to send new weights to workers
    save_model_interval = 200 # How often to save a checkpoint
    max_steps_per_episode = temp_env.state_space_size * 10
    
    # --- 3. Create Communication Queues ---
    # Workers send experience here
    experience_queue = multiprocessing.Queue(maxsize=num_workers * 5)
    # Learner sends updated model weights here
    model_weights_queue = multiprocessing.Queue()

    print(f"\n--- Starting Training with {num_workers} Workers ---")
    
    # --- 4. Start Worker Processes ---
    workers = []
    initial_weights = latest_weights
    for i in range(num_workers):
        model_weights_queue.put({'weights': latest_weights, 'epsilon': central_agent.epsilon})
        process = multiprocessing.Process(
            target=worker_task,
            args=(song_to_learn, experience_queue, model_weights_queue, i, episodes_per_worker, max_steps_per_episode)
        )
        process.start()
        workers.append(process)

    start_time = time.time()
    episodes_processed = 0
    training_steps = 0
    recent_rewards = deque(maxlen=100)
    total_loss = 0.0 

    try:
        # --- 5. Main Learning Loop ---
        while any(p.is_alive() for p in workers) or not experience_queue.empty():
            try:
                # Get experience from any worker that has finished an episode
                episode_experiences, total_reward = experience_queue.get(timeout=1.0)
                recent_rewards.append(total_reward)
            except:
                continue # If queue is empty, loop again
            
            # Add all experiences from that episode to the central agent's memory
            for experience in episode_experiences:
                central_agent.remember(*experience)
            
            episodes_processed += 1
            
            # Perform a training step if memory is sufficient
            if len(central_agent.memory) > central_agent.batch_size:
                central_agent.replay()
                training_steps += 1

                # Periodically send updated weights to the workers
                if training_steps % update_weights_interval == 0:
                    latest_weights = central_agent.model.get_weights()
                    # Clear the queue and add the new weights for workers to pick up
                    while not model_weights_queue.empty():
                        model_weights_queue.get()
                    for _ in range(num_workers):
                        model_weights_queue.put({'weights': latest_weights, 'epsilon': central_agent.epsilon})
                
                # Periodically save a checkpoint
                if training_steps > 0 and training_steps % save_model_interval == 0:
                    with open(CHECKPOINT_PATH, 'wb') as f:
                        pickle.dump({
                            'model_weights': central_agent.model.get_weights(),
                            'target_model_weights': central_agent.target_model.get_weights(),
                            'epsilon': central_agent.epsilon
                        }, f)
                    print(f"\n--- Checkpoint Saved at Step {training_steps} ---")


            if episodes_processed % 10 == 0 and len(recent_rewards) > 0:
                avg_reward = np.mean(recent_rewards)
                
                # <--- 3. LOG METRICS TO TENSORBOARD ---
                writer.add_scalar('Performance/Average Reward', avg_reward, episodes_processed)
                writer.add_scalar('Agent/Epsilon', central_agent.epsilon, episodes_processed)
                if training_steps > 0:
                    avg_loss = total_loss / training_steps
                    writer.add_scalar('Train/Loss', avg_loss, episodes_processed)
                print(f"Episodes: {episodes_processed}/{total_episodes} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {central_agent.epsilon:.3f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        # --- 6. Cleanup ---
        print("\n--- Training Finished or Interrupted: Cleaning up ---")
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()
        
        print("Saving final checkpoint...")
        with open(CHECKPOINT_PATH, 'wb') as f:
            pickle.dump({
                'model_weights': central_agent.model.get_weights(),
                'target_model_weights': central_agent.target_model.get_weights(),
                'epsilon': central_agent.epsilon
            }, f)
        
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds.")