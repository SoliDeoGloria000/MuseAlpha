import multiprocessing
import time
import sys
import os
from collections import deque
import numpy as np

from environment import PianoEnvironment
from agent import DQNAgent
from checkpoint_manager import save_checkpoint, load_checkpoint

def worker_task(song_string, experience_queue, model_weights_queue, worker_id, episodes_per_worker, max_steps):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    env = PianoEnvironment(song_string, max_steps_per_episode=max_steps)
    state_size = env.state_space_size
    action_size = env.action_space_size
    agent = DQNAgent(state_size, action_size) 
    
    try:
        initial_weights = model_weights_queue.get()
        agent.model.set_weights(initial_weights)
    except Exception:
        return

    for e in range(episodes_per_worker):
        try:
            new_weights = model_weights_queue.get_nowait()
            agent.model.set_weights(new_weights)
        except:
            pass 

        state = env.reset()
        done = False
        episode_experiences = []
        total_reward = 0
        
        while not done:
            action = agent.act(state) 
            next_state, reward, done = env.step(action)
            episode_experiences.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

        experience_queue.put((episode_experiences, total_reward))

if __name__ == "__main__":
    song_to_learn = "4|[ute]|s|[o1]|[wur]||4|[ute]||1|[wur]||4|[utfe]|h|[g1]|[wurf]|a|[p4]||[utea]|s|[o1]|[wur]||1|[wur]||4|[utfe]|h|[g1]|[wurf]|a|[p4]||[utea]|s|[o1]|[wur]||[a3]|[wur]||[f6]|[ute]||[y2]|[qe]||2|[tie]||1|[wt(]||[o5]||[ywqE]|P|[d1]|[ysqe]|p|[s1]|[yqeP]|p|[s1]|[wt^]||1|[wt^0]|s|[d1]|[wYED]|g|[h1]|[ywPE]|s|[d1]|[ysqe]|p|[s1]|[wt^9]||1|[wt^0]|s|[g2]|[yie]||[f3]|[wur]||[p6]|[utoe]|p|[a2]|[ysro]|d|[a2]|[wusro]|d|[u2]|[wt6]|[tie9]|[yqoPE5]||[utso851]|||4|[ute]|1|[wur]||4|[ute]||1|[wur]||4|[utfe]|h|[g1]|[wurf]|a|[p4]|[utea]|s|[o1]|[wur]||[u4]|[ute]||1|[wur]||4|[ute]||1|[wur]||4|[utfe]|h|[g1]|[wurf]|a|[p4]||[utea]|s|[o1]|[wur]||[a3]|[wur]||[f6]|[ute]||[y2]|[qe]||2|[tie]||1|[wt(]||[o5]|[ywqE]|P|[d1]|[ysqe]|p|[s1]|[yqeP8]|p|[s1]|[wt^9]||1|[wt^0]|s|[d1]|[wYED]|g|[h1]|[ywPE]|s|[d1]|[ysqe8]|p|[s1]|[wt^9]||1|[wt^0]|s|[g2]|[yie]||[D2]|[wtoY]||[p2]|[wYPE]|D|[d2]|[ysoE]|P|[d2]|[wsoYE]|P|[Y2]|[wt6]|[tie9]||[yqoPE5]|||[tsoY851]"

    print("--- Initializing Central Agent ---")
    temp_env = PianoEnvironment(song_to_learn)
    state_size = temp_env.state_space_size
    action_size = temp_env.action_space_size
    
    central_agent = DQNAgent(state_size, action_size)
    checkpoint_filename = "piano_agent_checkpoint.pkl"
    load_checkpoint(central_agent, checkpoint_filename)

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # --- MODIFICATION: Increased training duration ---
    total_episodes = 20000
    
    episodes_per_worker = total_episodes // num_workers
    batch_size = 64
    update_weights_interval = 10
    save_model_interval = 200
    max_steps_per_episode = state_size * 20 
    
    experience_queue = multiprocessing.Queue(maxsize=num_workers*2)
    model_weights_queue = multiprocessing.Queue()

    print(f"\n--- Starting Training with {num_workers} Workers ---")
    print(f"Total episodes to run: {total_episodes}")
    
    workers = []
    initial_weights = central_agent.model.get_weights()
    for i in range(num_workers):
        model_weights_queue.put(initial_weights)
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

    try:
        while any(p.is_alive() for p in workers) or not experience_queue.empty():
            try:
                episode_experiences, total_reward = experience_queue.get(timeout=1.0)
                recent_rewards.append(total_reward)
            except:
                continue
            
            for experience in episode_experiences:
                central_agent.remember(*experience)
            
            episodes_processed += 1
            
            if len(central_agent.memory) > batch_size:
                central_agent.replay(batch_size)
                training_steps += 1

                if training_steps % update_weights_interval == 0:
                    latest_weights = central_agent.model.get_weights()
                    for _ in range(num_workers): 
                        if not model_weights_queue.full():
                            model_weights_queue.put(latest_weights, block=False)
                
                if training_steps > 0 and training_steps % save_model_interval == 0:
                    save_checkpoint(central_agent, checkpoint_filename)

            if episodes_processed % 10 == 0 and len(recent_rewards) > 0:
                avg_reward = np.mean(recent_rewards)
                print(f"Episodes: {episodes_processed} | Avg Reward (last 100): {avg_reward:.2f} | "
                      f"Memory: {len(central_agent.memory)} | Epsilon: {central_agent.epsilon:.3f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        print("\n--- Training Finished or Interrupted: Cleaning up ---")
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()
        
        print("Saving final checkpoint...")
        save_checkpoint(central_agent, checkpoint_filename)
        
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds.")