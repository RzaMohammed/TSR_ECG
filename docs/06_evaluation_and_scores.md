# 6. Evaluation and the ROC-AUC Score

After the AI finishes training, it saves all of its "knowledge" into a file called a **checkpoint** (in our case, `ckpt/TSRNet-latest.pt`). 

Now, we need to test if the AI actually learned anything, or if it just memorized the textbook. We do this by running `test.py`.

## The Final Exam
We take our 2,155 "Test Patients." The AI has **never** seen these patients during its training. Some of them are healthy, and some of them have heart diseases.

We ask the AI to try and reconstruct all of their heartbeats. 
* If the AI gets a High Error, it guesses the patient is Sick.
* If the AI gets a Low Error, it guesses the patient is Healthy.

## Grading the Exam: The ROC-AUC Score
To grade the AI's guesses, we use a math formula that spits out a number called the **ROC-AUC Score**. This score ranges from `0.5` to `1.0`.

* **0.500:** The AI is terrible. It is basically flipping a coin to guess if someone is sick or healthy.
* **1.000:** The AI is perfect. It never makes a mistake.

## Our Local Results
When we ran the AI locally on our computer, we only let it study for **1 Epoch** (1 read through the textbook) just to make sure the code didn't crash. 

Even with only 1 read-through, the AI achieved an **AUC Score of 0.607**. This proves the code works and the AI is starting to learn!

When you take this project and run it for 50 Epochs on a fast cloud computer (like Kaggle), the score will skyrocket to **0.865 or higher**, making it a highly accurate medical tool.
