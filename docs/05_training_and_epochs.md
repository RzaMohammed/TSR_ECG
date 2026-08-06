# 5. Training, Epochs, and the Range Error

Once the data is preprocessed, we run `train.py` to start teaching the AI.

## What is an Epoch?
Imagine the AI is a student, and the 14,241 healthy patient heartbeats are a textbook.
* Reading through the entire textbook **once** is called **1 Epoch**.
* Reading it through twice is **2 Epochs**.

The more epochs the AI completes, the better it gets at its job. For this project to reach its maximum potential, it usually needs to run for about 50 Epochs on a powerful GPU computer.

## The "Range Error" (A Bug We Fixed!)
When we tried to test the code locally on our computer by asking it to run just **1 Epoch**, we noticed a huge problem. It finished Epoch 0, and then immediately started running Epoch 1! It was doing double the work we asked for.

**Why did this happen?**
The original creator of this repository wrote a tiny bug in the code on line 66 of `train.py`:
```python
# The original buggy code:
for epoch in range(0, args.epochs + 1):
```

**Understanding the Bug:**
In programming, computers start counting at `0`, not `1`. 
If we asked for `1` epoch, the code did math: `1 + 1 = 2`. It then told the computer to run a loop in the `range(0, 2)`. 
In Python, `range(0, 2)` means it will run for the number `0`, and then run again for the number `1`. That's **two full reads** of the textbook!

**The Fix:**
We went into the code and deleted the `+ 1`.
```python
# The fixed code:
for epoch in range(0, args.epochs):
```
Now, if you ask for `1` epoch, the computer calculates `range(0, 1)`, which means it only runs for the number `0` and then successfully stops. Bug fixed!

---
**Next up:** [Evaluation and the ROC-AUC Score](06_evaluation_and_scores.md)
