# Path Estimation with Kalman Filter

This project is designed to help understand how the Kalman filter works in a simple scenario. It demonstrates how to estimate and smooth the path of a moving object based on noisy tracking points (dots). By applying the Kalman filter, we can predict and correct the object's trajectory over time, resulting in a more reliable path, even when the input data is imperfect or noisy.

## Circuit Diagram

Here is the circuit diagram used in the project:

![Circuit Diagram](circuit.jpg)

## Input Data: Noisy Tracking Points

The input consists of noisy tracking points (dots) representing the object's position over time. These points may appear imprecise or erratic:

![Noisy Tracking Points](result/circuit_initilize_posistion.png)

## Path Estimation Result

After applying the Kalman filter to the noisy tracking points, the result is a smooth and accurate estimated path:

![Kalman Filter Path Estimation](result/KF_Estimation.png)

## Features:

- **Input**: A series of noisy tracking points (dots).
- **Kalman Filter**: Used to smooth the data and estimate the true path.
- **Output**: A continuous and accurate trajectory.

## How It Works:

1. **Input Data**: A set of dots representing the object's positions at different times.
2. **Kalman Filter**: Predicts the next position based on previous data, then corrects it when new measurements are received.
3. **Path Output**: The Kalman filter generates a smoothed, continuous path that accurately reflects the object's movement.

## Applications:

- Robotics
- Computer Vision
- Autonomous Vehicles
