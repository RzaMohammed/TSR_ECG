# 2. In-Depth Guide: What is AI Anomaly Detection?

When you hear about Medical AI, you usually hear about "Classification." But our project uses **"Anomaly Detection"** (specifically, **Restoration-Based Anomaly Detection**). 

Why did we choose this approach, and how does it mathematically work? This document breaks down the deep learning theory in simple, project-specific terms.

---

## Part 1: The Problem with Traditional AI (Supervised Learning)

Normally, AI is trained using **Supervised Learning**. 
Imagine teaching a child the difference between cats and dogs. You show them 10,000 pictures of cats and 10,000 pictures of dogs. Eventually, the child learns the patterns and can classify a new picture.

### Why this fails in Cardiology
If we wanted to use Supervised Learning for ECGs, we would need 10,000 examples of *every single heart disease*. 
1. **Class Imbalance:** We have millions of healthy ECGs, but very few examples of rare diseases (like Brugada Syndrome). The AI becomes biased and starts guessing "Healthy" for everyone just to score high on average.
2. **Novel Diseases:** If a patient has a brand new type of heart defect that the AI has never seen before, the AI will confidently misdiagnose them because it only knows the diseases it was explicitly taught.

---

## Part 2: The Solution (Unsupervised Anomaly Detection)

Instead of teaching the AI every disease in existence, we use **Anomaly Detection**. We only teach the AI one thing: **What does a perfectly normal, healthy heart look like?**

If the AI becomes an absolute master of "Normal", then *anything* that strays from normal is flagged as an "Anomaly" (a disease). The AI doesn't need to know the name of the disease; it just knows the heart is sick.

### How do we teach it "Normal"? (The Restoration Approach)
TSRNet uses a concept similar to an **Autoencoder**. 
During training, we give the AI a healthy ECG, but we sabotage it. We literally delete (mask) 30% of the numbers in the data array. We replace the healthy voltages with zeros.

We then force the AI to reconstruct the missing data. 
Because the AI only ever practices on healthy hearts, its "brain" is physically hardwired to draw healthy P-QRS-T waves. 

---

## Part 3: The Mathematics of Detection (Calculating the Error)

When a real patient comes into the hospital, how does the AI actually decide if they are sick? 

1. We take the new patient's 12-lead ECG.
2. We mask out 30% of their signal.
3. We ask the AI to fill in the blanks.
4. We compare the AI's drawing to the real 30% we hid, and we calculate the **Reconstruction Error**.

### The Mean Squared Error (MSE) Formula
To calculate the difference between the Real Signal and the AI's Fake Signal, the code uses a math formula called MSE.

```math
Error = (Real\_Voltage - Predicted\_Voltage)^2
```

**Example 1: A Healthy Patient (Low Error)**
* Real Voltage at time step 500: `0.85`
* AI's Prediction at time step 500: `0.84`
* Difference: `0.01`
* Squared Error: `0.0001` (Tiny error! The patient is healthy).

**Example 2: A Sick Patient (High Error)**
Let's say the patient has a heart attack, causing a massive, abnormal spike in their T-wave.
* Real Voltage at time step 500: `0.90`
* AI's Prediction: `0.10` (Because the AI expects a normal, flat T-wave here).
* Difference: `0.80`
* Squared Error: `0.64` (Massive error! The patient is flagged as sick).

### Project-Specific Output
When you run `test.py` on your computer, the script processes all 2,155 patients in the test set. 
Inside the code, it creates an array of 2,155 "Anomaly Scores". 

```python
# Example of what the AI outputs internally:
anomaly_scores = [0.002, 0.015, 0.893, 0.004, 0.991, ...]
```
The computer then sets a **Threshold** (like a passing grade on a test). For example, if the threshold is `0.5`:
* Patient 1 (`0.002`) -> Healthy
* Patient 3 (`0.893`) -> Sick

By using this Restoration Error method, TSRNet can detect *any* heart disease, even ones it has never seen before, simply because sick hearts cannot be smoothly reconstructed by a healthy-trained AI.
