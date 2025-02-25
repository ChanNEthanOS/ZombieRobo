import os
import pandas as pd

class Memory:
    def __init__(self, filepath="memory/memory.pkl"):
        self.file = filepath

    def log_action(self, state, action, reward):
        entry = {'state': state, 'action': action, 'reward': reward}
        df = self.load_memory()
        df = df.append(entry, ignore_index=True)
        df.to_pickle(self.file)

    def load_memory(self):
        if os.path.exists(self.file):
            return pd.read_pickle(self.file)
        else:
            return pd.DataFrame(columns=['state', 'action', 'reward'])
