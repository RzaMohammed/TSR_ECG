# 1. What is an ECG and the PTB-XL Dataset?

## What is an ECG?
ECG stands for **Electrocardiogram**. Every time your heart beats, it generates a tiny electrical signal. An ECG is simply a recording of that electrical signal over time. It looks like a squiggly line on a piece of paper or a monitor.

Doctors look at these squiggly lines to see if a heart is healthy or if someone has a disease, like an irregular heartbeat (arrhythmia) or a heart attack.

## What is a "12-Lead" ECG?
When you go to a hospital for a serious heart check, the doctor doesn't just put one sticker on your chest. They put **12 different stickers** on your chest, arms, and legs. Each sticker is called a **"lead"**.

Because they are in different spots, each sticker records the electrical signal from a slightly different angle. So, instead of getting just 1 squiggly line for a heartbeat, we get **12 lines** at the same time. This gives the doctor (and our AI) a complete 3D picture of what the heart is doing.

## What is the PTB-XL Dataset?
To teach an AI to recognize heart diseases, it needs to study thousands of examples. 

We used a public database called **PTB-XL**. It is a massive, free library that contains the 12-lead ECG recordings of over 21,000 real patients. 

These files are stored in a format called **WFDB (Waveform Database)**. This is just a standard computer format used by hospitals to save the digital heartbeat recordings (saved as `.dat` files) alongside the patient's information.

---
**Next up:** [What is AI Anomaly Detection?](02_what_is_ai_anomaly_detection.md)
