# challenge_generator.py
import random
import time

CHALLENGES = [
    "Blink twice",
    "Turn your head left",
    "Turn your head right",
    "Look up",
    "Look down",
    "Say hello"
]

def generate_challenge():
    challenge = random.choice(CHALLENGES)
    print("CHALLENGE:", challenge)
    return challenge

if __name__ == "__main__":
    c = generate_challenge()
    print(c)