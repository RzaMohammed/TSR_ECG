# 4. The TSRNet Architecture (How the AI works)

This is the most complex part of the project, but we can understand it using a simple analogy.

Our AI model is called **TSRNet**. It learns to detect heart diseases using two main tricks: **The Masking Puzzle** and having **Two Brains**.

## Trick 1: The Masking Puzzle
Imagine we have a photo of a healthy heart. We take a pair of scissors and cut out 30% of the photo (we call this **masking**). 

We give the incomplete photo to the AI and ask it, *"Draw what you think the missing pieces look like."*
Because the AI practices doing this thousands of times on *healthy* hearts, it becomes an absolute expert at perfectly re-drawing a healthy heartbeat.

Later, if we give it a sick heartbeat with missing pieces, the AI will try to fill in the blanks with *healthy* patterns. The result will look totally different from the real sick heartbeat. That massive difference (Error) is how it detects a disease!

## Trick 2: The Two Brains (Time & Spectrogram)
A human doctor looks at an ECG as a squiggly line over time. But computers can look at data in multiple ways at once.

TSRNet has two separate "brains" (called Encoders) that look at the heartbeat from two different perspectives:
1. **The Time Brain:** This looks at the normal squiggly line. It is great at seeing sudden drops or spikes in the heartbeat.
2. **The Spectrogram Brain (Frequency):** This brain mathematically converts the squiggly line into a colorful 2D image called a Spectrogram (or heat map). This map shows the "pitch" and "energy" of the heartbeat. 

By having two brains, the AI gets a much deeper understanding of the heart. 

## Putting it together (Cross-Attention)
The two brains need to talk to each other to agree on a final answer. They do this using a mechanism called **Cross-Attention**. 
Think of it as the two brains having a conversation:
* *Time Brain:* "Hey, I see a weird spike right here!"
* *Spectrogram Brain:* "Let me check the energy levels at that exact moment... Yes, you are right!"

Together, they reconstruct the missing pieces of the puzzle and output their final prediction.

---
**Next up:** [Training, Epochs, and the Range Error](05_training_and_epochs.md)
