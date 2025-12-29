# behavior_liveness.py
import time
import numpy as np

def get_behavior_score():
    prompts = ["Say your name", "Say the number five", "Say hello"]
    response_times = []

    for prompt in prompts:
        print(prompt)
        start = time.time()
        input()
        response_times.append(time.time() - start)

    response_times = np.array(response_times)
    variability = np.std(response_times)
    mean_time = np.mean(response_times)

    if 0.3 < mean_time < 3.0 and variability > 0.1:
        score = 1.0
    elif mean_time < 5.0:
        score = 0.6
    else:
        score = 0.0

    return float(np.clip(score, 0, 1))

if __name__ == "__main__":
    print(get_behavior_score())