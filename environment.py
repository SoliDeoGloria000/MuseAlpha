# environment_v2.py
import numpy as np
from collections import deque
from music_parser import parse_sheet_music

class PianoEnvironment:
    def __init__(self, song_string, max_steps=2000):
        self.song_string = song_string
        self.parsed_song = parse_sheet_music(song_string)
        self.state_space_size = len(self.parsed_song)
        self.max_steps = max_steps
        
        # Build action space from unique actions in song
        self.actions, self.action_to_id = self._build_action_space()
        self.action_space_size = len(self.actions)
        
        # Enhanced rewards
        self.R_CORRECT = 5.0
        self.R_WRONG = -5.0
        self.R_STUCK = -15.0
        self.R_FINISH = 1000.0
        self.R_PROGRESS = 20.0  # Bonus for making progress
        
        self.reset()
    
    def _canonical_key(self, action):
        if action['type'] == 'chord':
            return ('chord', tuple(sorted(action['notes']))) # This is correct
        elif action['type'] == 'note':
            # This was the bug. It should be action['note'], not action['notes']
            return ('note', action['note'])
        else:
            return ('pause', action.get('duration', 1))

    def _build_action_space(self):
        unique_actions = []
        seen = set()
        for action in self.parsed_song:
            key = self._canonical_key(action)
            if key not in seen:
                seen.add(key)
                # This logic also needs to handle single notes correctly
                if action['type'] == 'chord':
                    # This part is correct
                    unique_actions.append({'type':'chord','notes':sorted(action['notes'])})
                else:
                    # This correctly appends single notes and pauses
                    unique_actions.append(action)
        action_to_id = {self._canonical_key(a): i for i, a in enumerate(unique_actions)}
        return unique_actions, action_to_id
    
    def reset(self):
        """Resets the environment to the initial state."""
        self.current_step = 0
        self.total_steps = 0
        self.wrong_streak = 0
        self.history = deque(maxlen=10)  # Track recent actions
        return self._get_state()
    
    def _get_state(self):
        """Constructs the state vector with positional and progress information."""
        # State size = song length + context window + progress + wrong streak
        state = np.zeros(self.state_space_size + 12, dtype=np.float32)
        
        # One-hot encode the current position in the song
        if self.current_step < self.state_space_size:
            state[self.current_step] = 1.0
        
        # Add context: preview of the next few actions' types
        for i in range(1, 4):
            if self.current_step + i < self.state_space_size:
                next_action = self.parsed_song[self.current_step + i]
                # Encode type: note=0.3, chord=0.6, pause=0.9
                if next_action['type'] == 'note':
                    state[self.state_space_size + i] = 0.3
                elif next_action['type'] == 'chord':
                    state[self.state_space_size + i] = 0.6
                else:  # pause
                    state[self.state_space_size + i] = 0.9
        
        # Add progress indicator
        state[self.state_space_size + 10] = self.current_step / max(1, self.state_space_size)
        
        # Add wrong streak indicator (helps agent learn to not get stuck)
        state[self.state_space_size + 11] = min(1.0, self.wrong_streak / 10.0)
        
        return state
    
    def step(self, action_id):
        """
        Executes one time step within the environment.
        
        Returns:
            state (np.array): The new state.
            reward (float): The reward for the action.
            done (bool): Whether the episode has ended.
        """
        done = False
        reward = 0.0
        
        if self.current_step >= self.state_space_size:
            return self._get_state(), 0.0, True
        
        # Get the correct action for the current step
        correct_action = self.parsed_song[self.current_step]
        correct_key = self._canonical_key(correct_action)
        correct_id = self.action_to_id.get(correct_key, -1)
        
        if action_id == correct_id:
            self.current_step += 1
            
            # Base reward for correct action
            reward = self.R_CORRECT
            
            # Progress bonus at key milestones
            progress_pct = self.current_step / self.state_space_size
            if progress_pct > 0.75:
                reward += self.R_PROGRESS * 2
            elif progress_pct > 0.5:
                reward += self.R_PROGRESS
            elif progress_pct > 0.25:
                reward += self.R_PROGRESS
            
            self.wrong_streak = 0
            
            # Check if the song is finished
            if self.current_step >= self.state_space_size:
                reward += self.R_FINISH
                done = True
        else:
            reward = self.R_WRONG
            self.wrong_streak += 1
            
            # Penalty scales with how long the agent is stuck
            if self.wrong_streak > 5:
                reward -= self.wrong_streak * 0.5
            
            if self.wrong_streak >= 20:
                reward += self.R_STUCK
                done = True
        
        self.total_steps += 1
        
        # End episode if it runs for too long
        if self.total_steps >= self.max_steps:
            done = True
        
        return self._get_state(), reward, done

    def get_action_mask(self):
        """
        Returns a mask where only the correct action for the current step is valid.
        This can be used to guide the agent's learning process.
        """
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        if self.current_step < self.state_space_size:
            correct_action = self.parsed_song[self.current_step]
            correct_key = self._canonical_key(correct_action)
            action_idx = self.action_to_id.get(correct_key)
            if action_idx is not None:
                mask[action_idx] = 1.0
        else:
            # If the episode is over, all actions could be considered valid or invalid
            # A mask of all zeros is often safest.
            pass
        return mask
