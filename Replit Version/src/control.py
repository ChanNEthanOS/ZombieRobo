from src.ai_model import AIAgent, RLModel

class FrankControl:
    def __init__(self, state):
        self.state = state
        self.controlled_by_frank = True
        self.key_mapping = {
            'w': 'move_up',
            's': 'move_down',
            'a': 'move_left',
            'd': 'move_right',
            'space': 'shoot',
        }
        self.ai_agent = AIAgent(RLModel())

    def ai_move(self):
        action = self.ai_agent.advise(self.state)
        self.state = self.state.apply_action(self.state.get_possible_actions()[action])
        return self.state

    def control(self, user_input):
        if not self.controlled_by_frank:
            return self.ai_move()

        action = self.key_mapping.get(user_input.lower())
        if action:
            print(f"\n> Action: {action.upper()}", flush=True)
            self.state = self.state.apply_action(action)
            print(str(self.state), flush=True)
        else:
            print("\nInvalid command. Use W/A/S/D/Space.", flush=True)

        return self.state