import numpy as np
import os

def generate():
    os.makedirs('data', exist_ok=True)
    t = np.linspace(0, 10, 5000)
    base = np.sin(2 * np.pi * 1.2 * t)
    peaks = np.zeros(5000)
    for i in range(50, 5000, 416):  # Approx 72 bpm
        peaks[i:i+10] = np.linspace(0, 5, 10)
        peaks[i+10:i+20] = np.linspace(5, 0, 10)
    ecg = base + peaks
    
    N = 4
    train_data = np.zeros((N, 5000, 12))
    for n in range(N):
        for i in range(12):
            train_data[n, :, i] = ecg + np.random.normal(0, 0.1, 5000)
    
    test_data = np.copy(train_data)
    # Give the anomaly (label 1) a distinct signal to ensure scores vary
    for i in range(12):
        test_data[0, :, i] += np.random.normal(0, 1.0, 5000) 
    
    labels = np.zeros(N)
    labels[0] = 1
    
    np.save('data/train.npy', train_data)
    np.save('data/test.npy', test_data)
    np.save('data/label.npy', labels)
    print("Dummy data generated.")

if __name__ == '__main__':
    generate()
