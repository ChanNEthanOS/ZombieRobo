# ai_game_project/ai_corporation/AICorporation.py

from ai_agent import AIAgentAI

class AICorporation:
    def __init__(self, state):
        self.departments = {
            "strategy": [AIAgentAI(i) for i in range(100)],
            "tactics": [AIAgentAI(i + 100) for i in range(200)],
            "execution": [AIAgentAI(i + 300) for i in range(200)]
        }
        self.state = state

    def execute_strategy(self):
        for department, agents in self.departments.items():
            for agent in agents:
                action = agent.decide_action(self.state)
                self.state = self.state.apply_action(action)
        return self.state
