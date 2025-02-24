import torch
import torch.nn as nn

class RLModel(nn.Module):
    def __init__(self):
        super(RLModel, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AIAgent:
    def __init__(self, model):
        self.model = model

    def advise(self, state):
        state_tensor = torch.tensor([state.player_position], dtype=torch.float32)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
