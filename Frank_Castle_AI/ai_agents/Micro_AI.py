from models.mistral_model import generate_decision

def run():
    prompt = "Micro_AI, decide next action: navigate map, buy door, reload, or shoot?"
    decision = generate_decision(prompt)
    print("Micro_AI decision:", decision)

if __name__ == "__main__":
    run()
