# agent_v2.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
import random
from collections import deque

class ImprovedDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.prioritized_memory = deque(maxlen=1000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.lr = 0.0001
        self.batch_size = 64
        self.target_update_freq = 100
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        self.train_steps = 0
    
    def _build_model(self):
        """Deeper network with dropout for better generalization"""
        inp = layers.Input(shape=(self.state_size,))
        
        x = layers.Dense(512, activation='relu')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        value = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1)(value)
        
        advantage = layers.Dense(64, activation='relu')(x)
        advantage = layers.Dense(self.action_size)(advantage)
        
        mean_advantage = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        out = value + (advantage - mean_advantage)
        
        model = models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.lr),
            loss=losses.Huber()
        )
        return model
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done, next_mask=None):
        self.memory.append((state, action, reward, next_state, done, next_mask))
        if reward > 10:
            self.prioritized_memory.append((state, action, reward, next_state, done, next_mask))
    
    def act(self, state, mask=None):
        """
        Chooses an action.
        - With probability epsilon, it takes a random action (exploration).
        - Otherwise, it takes the best known action (exploitation).
        """
        # --- THIS IS THE CORRECTED LOGIC ---
        # Exploration: Choose a completely random action from ALL possible actions.
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: Use the learned policy (Q-values) to decide.
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        
        # Apply the mask during exploitation to ensure the chosen action is valid.
        if mask is not None:
            # Set Q-value of invalid actions to negative infinity so they are never chosen.
            q_values[mask < 0.5] = -np.inf 
            
        return int(np.argmax(q_values))

    def replay(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return 0.0

        regular_size = int(batch_size * 0.7)
        priority_size = batch_size - regular_size

        minibatch = []
        if len(self.memory) >= regular_size:
            minibatch.extend(random.sample(self.memory, regular_size))
        
        if len(self.prioritized_memory) >= priority_size:
            minibatch.extend(random.sample(self.prioritized_memory, priority_size))
        else:
            minibatch.extend(random.sample(self.memory, priority_size))

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch], dtype=np.float32)
        next_masks = np.array([m[5] if len(m) > 5 and m[5] is not None else np.ones(self.action_size, np.float32) for m in minibatch])

        q_current = self.model.predict(states, verbose=0)
        q_next_online = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        for i in range(len(minibatch)):
            q_next_online[i, next_masks[i] < 0.5] = -np.inf
        
        best_actions = np.argmax(q_next_online, axis=1)
        target_values = rewards + (1.0 - dones) * self.gamma * q_next_target[np.arange(len(minibatch)), best_actions]

        q_target = q_current.copy()
        q_target[np.arange(len(minibatch)), actions] = target_values

        history = self.model.fit(states, q_target, epochs=1, verbose=0, batch_size=32)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss

