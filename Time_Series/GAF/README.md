# **GAF Implementation + Mini Project**

## Introduction  
This project implements the **Gramian Angular Field (GAF)** technique for encoding time series data as images. GAF is useful for converting 1D signals into 2D representations, enabling the use of image-based models (e.g., CNNs) for time-series classification or anomaly detection. The project supports both **GASF (Summation)** and **GADF (Difference)** variants, along with basic visualization.

---

## Core Algorithm  
1. **Min-Max Normalization**  
   The input signal is normalized to the range \([-1, 1]\):  
   \[
   X' = 2 \times \frac{X - \min(X)}{\max(X) - \min(X)} - 1
   \]

2. **Polar Encoding**  
   Convert normalized values to angles using:  
   \[
   \phi_i = \arccos(X'_i)
   \]

3. **GAF Construction**  
   - **GASF**:  
     \[
     G[i,j] = \cos(\phi_i + \phi_j)
     \]
   - **GADF**:  
     \[
     G[i,j] = \cos(\phi_i - \phi_j)
     \]

4. **Visualization**  
   Each GAF matrix is visualized using Matplotlib's `imshow()` for intuitive inspection.
