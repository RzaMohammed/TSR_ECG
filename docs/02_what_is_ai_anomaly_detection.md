# 2. What is AI Anomaly Detection?

## The Problem
Normally, to teach an AI to recognize a heart disease, you have to show it thousands of examples of healthy hearts, AND thousands of examples of sick hearts (like heart attacks, murmurs, etc.). 
However, there are hundreds of different ways a heart can be sick, making it really hard to find enough examples for every single disease.

## The Anomaly Detection Solution
Instead of trying to teach the AI every possible disease, **Anomaly Detection** takes a smarter approach: **We only teach the AI what a perfectly healthy heart looks like.**

During training, we force the AI to memorize the exact patterns, rhythms, and shapes of a normal, healthy heartbeat. It becomes a master of "normal".

## How does it detect a disease?
Once the AI is fully trained on healthy hearts, we show it a brand new patient's heartbeat and ask it to recreate it.

* If the patient is healthy, the heartbeat looks familiar to the AI. The AI easily recreates it with **Low Error**.
* If the patient is sick, the heartbeat looks weird and strange to the AI. Because the AI has never seen it before, it does a terrible job recreating it, resulting in a **High Error**.

If the Error is high enough, the system flags the heartbeat as an **Anomaly** (a fancy word for something abnormal or sick). We don't need to know *what* disease it is; we just know it is definitely not healthy!

---
**Next up:** [Data Preprocessing (Cleaning the Data)](03_data_preprocessing_explained.md)
