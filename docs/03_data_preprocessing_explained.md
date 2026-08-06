# 3. Data Preprocessing (Cleaning the Data)

Before we can hand the hospital data over to the AI, we have to prepare it. In our code, this step is handled by `preprocess.py`.

## Why is hospital data "messy"?
When a nurse records a patient's heartbeat, the signal is rarely perfect. It picks up a lot of "noise":
1. **Breathing:** As the patient breathes in and out, their chest moves. This causes the entire heartbeat line to slowly drift up and down (called Baseline Wander).
2. **Muscle Twitches:** If the patient shivers or moves, it causes jagged spikes on the graph.
3. **Electrical Interference:** The hospital room is full of lights and machines plugged into the walls. These machines leak tiny amounts of electricity that the ECG sensors accidentally pick up, adding "fuzz" to the line.

If we give this messy, fuzzy, wandering line to the AI, it will get very confused and might think the noise is a heart disease!

## How we clean it (Filtering)
In `preprocess.py`, we apply digital **Filters** to clean the noise, much like a water filter cleans dirt:
* **High-Pass Filter:** This acts like a bouncer that only lets fast signals through. It blocks the slow, rolling waves caused by breathing.
* **Low-Pass & Notch Filters:** These act like bouncers that block super-fast, fuzzy signals caused by hospital electricity and muscle twitches.

## Normalization (Making it fair)
Everyone's heart beats at slightly different electrical strengths. Some are tall on the graph, some are short. 
To make it fair for the AI, we **Normalize** the data. This is a mathematical trick that stretches or squishes everyone's heartbeat so they all fit perfectly between the numbers `-1` and `1`. 

Once the data is clean and normalized, `preprocess.py` saves it into a special computer file format called NumPy (`.npy`). The AI can load these files extremely fast when it starts training.

---
**Next up:** [The TSRNet Architecture (How the AI works)](04_the_tsrnet_architecture.md)
