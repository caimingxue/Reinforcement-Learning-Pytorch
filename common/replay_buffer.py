import random
from collections import deque
class ReplayBuffer(object):
    def __init__(self, capacity):
        """Create Replay buffer.
       Parameters
       ----------
       size: int
           Max number of transitions to store in the buffer. When the buffer
           overflows the old memories are dropped.
       """
        self.buffer = deque(maxlen=capacity)
    def push(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)
