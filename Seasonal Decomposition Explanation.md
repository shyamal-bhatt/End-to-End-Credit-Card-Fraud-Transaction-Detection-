Below is an illustrative example that walks you through the calculations involved in classical additive seasonal decomposition using a small dataset of 10 records. In this example, we’ll assume a seasonal period \(p=3\) (i.e., the seasonal cycle repeats every 3 time units) for simplicity. We’ll show how to derive the Trend, Seasonal, and Residual components from the observed transaction amounts \(Y_t\).

> **Note:** In practice you might use a longer window (e.g., 30 days) but using \(p=3\) makes the step-by-step math easier to follow.

---

### 1. Example Data

Suppose we have the following 10 daily transaction amounts:

| Day | Observed \(Y_t\) |
|-----|------------------|
| 1   | 100              |
| 2   | 110              |
| 3   | 90               |
| 4   | 105              |
| 5   | 115              |
| 6   | 95               |
| 7   | 102              |
| 8   | 112              |
| 9   | 92               |
| 10  | 107              |

---

### 2. Estimating the Trend Component \(T_t\)

A common method is to use a moving average. Here, we use a centered moving average with a window size of 3. (For boundary points, one might use different techniques, but for this illustration we assume approximate values.)

For example, for Day 2–Day 9 we calculate:

- **Day 2 Trend:**  
  \( T_2 = \frac{Y_1 + Y_2 + Y_3}{3} = \frac{100 + 110 + 90}{3} = 100 \)

- **Day 3 Trend:**  
  \( T_3 = \frac{Y_2 + Y_3 + Y_4}{3} = \frac{110 + 90 + 105}{3} \approx 101.67 \)

- **Day 4 Trend:**  
  \( T_4 = \frac{Y_3 + Y_4 + Y_5}{3} = \frac{90 + 105 + 115}{3} \approx 103.33 \)

- **Day 5 Trend:**  
  \( T_5 = \frac{Y_4 + Y_5 + Y_6}{3} = \frac{105 + 115 + 95}{3} = 105 \)

- **Day 6 Trend:**  
  \( T_6 = \frac{Y_5 + Y_6 + Y_7}{3} = \frac{115 + 95 + 102}{3} \approx 104 \)

- **Day 7 Trend:**  
  \( T_7 = \frac{Y_6 + Y_7 + Y_8}{3} = \frac{95 + 102 + 112}{3} \approx 103 \)

- **Day 8 Trend:**  
  \( T_8 = \frac{Y_7 + Y_8 + Y_9}{3} = \frac{102 + 112 + 92}{3} \approx 102 \)

- **Day 9 Trend:**  
  \( T_9 = \frac{Y_8 + Y_9 + Y_{10}}{3} = \frac{112 + 92 + 107}{3} \approx 103.67 \)

For boundary days (Day 1 and Day 10), assume we approximate:
- **Day 1 Trend:** \( T_1 \approx 100 \)  
- **Day 10 Trend:** \( T_{10} \approx 107 \)

---

### 3. Detrending the Series

Next, subtract the trend from the observed values to obtain the detrended series \(D_t\):

\[
D_t = Y_t - T_t
\]

| Day | \(Y_t\) | \(T_t\)   | \(D_t = Y_t - T_t\)  |
|-----|---------|-----------|----------------------|
| 1   | 100     | 100       | \(100 - 100 = 0\)    |
| 2   | 110     | 100       | \(110 - 100 = 10\)   |
| 3   | 90      | 101.67    | \(90 - 101.67 \approx -11.67\) |
| 4   | 105     | 103.33    | \(105 - 103.33 \approx 1.67\)  |
| 5   | 115     | 105       | \(115 - 105 = 10\)   |
| 6   | 95      | 104       | \(95 - 104 = -9\)    |
| 7   | 102     | 103       | \(102 - 103 = -1\)   |
| 8   | 112     | 102       | \(112 - 102 = 10\)   |
| 9   | 92      | 103.67    | \(92 - 103.67 \approx -11.67\) |
| 10  | 107     | 107       | \(107 - 107 = 0\)    |

---

### 4. Estimating the Seasonal Component \(S_t\)

Assume a seasonal period \(p=3\). We group the detrended values by their position in the seasonal cycle:

- **Cycle Position 1:** Days 1, 4, 7, 10  
  \( S_1 = \text{Average}(D_1, D_4, D_7, D_{10}) = \frac{0 + 1.67 + (-1) + 0}{4} \approx 0.17 \)

- **Cycle Position 2:** Days 2, 5, 8  
  \( S_2 = \text{Average}(D_2, D_5, D_8) = \frac{10 + 10 + 10}{3} = 10 \)

- **Cycle Position 3:** Days 3, 6, 9  
  \( S_3 = \text{Average}(D_3, D_6, D_9) = \frac{-11.67 + (-9) + (-11.67)}{3} \approx -10.78 \)

Now, assign each day its seasonal factor based on its cycle position:

| Day | Cycle Position | Seasonal Factor \(S_t\) |
|-----|----------------|-------------------------|
| 1   | 1              | 0.17                    |
| 2   | 2              | 10                      |
| 3   | 3              | -10.78                  |
| 4   | 1              | 0.17                    |
| 5   | 2              | 10                      |
| 6   | 3              | -10.78                  |
| 7   | 1              | 0.17                    |
| 8   | 2              | 10                      |
| 9   | 3              | -10.78                  |
| 10  | 1              | 0.17                    |

---

### 5. Calculating the Residual Component \(R_t\)

The residual is what’s left after removing both the trend and seasonal components from the observed value:

\[
R_t = Y_t - T_t - S_t \quad \text{or} \quad R_t = D_t - S_t
\]

| Day | \(D_t\)           | Seasonal \(S_t\) | Residual \(R_t = D_t - S_t\)  |
|-----|-------------------|------------------|-------------------------------|
| 1   | 0.00              | 0.17             | \(0.00 - 0.17 = -0.17\)        |
| 2   | 10.00             | 10               | \(10.00 - 10 = 0\)             |
| 3   | -11.67            | -10.78           | \(-11.67 - (-10.78) \approx -0.89\) |
| 4   | 1.67              | 0.17             | \(1.67 - 0.17 = 1.50\)         |
| 5   | 10.00             | 10               | \(10.00 - 10 = 0\)             |
| 6   | -9.00             | -10.78           | \(-9 - (-10.78) \approx 1.78\) |
| 7   | -1.00             | 0.17             | \(-1.00 - 0.17 = -1.17\)       |
| 8   | 10.00             | 10               | \(10.00 - 10 = 0\)             |
| 9   | -11.67            | -10.78           | \(-11.67 - (-10.78) \approx -0.89\) |
| 10  | 0.00              | 0.17             | \(0.00 - 0.17 = -0.17\)        |

---

### 6. Summary Table of Calculations

Below is a combined table showing each step for the 10 sample records:

| Day | Observed \(Y_t\) | Trend \(T_t\) | Detrended \(D_t = Y_t - T_t\) | Cycle Position | Seasonal Factor \(S_t\) | Residual \(R_t = D_t - S_t\) |
|-----|------------------|---------------|-----------------------------|----------------|-------------------------|------------------------------|
| 1   | 100              | 100           | 0.00                        | 1              | 0.17                    | -0.17                        |
| 2   | 110              | 100           | 10.00                       | 2              | 10.00                   | 0.00                         |
| 3   | 90               | 101.67        | -11.67                      | 3              | -10.78                  | -0.89                        |
| 4   | 105              | 103.33        | 1.67                        | 1              | 0.17                    | 1.50                         |
| 5   | 115              | 105           | 10.00                       | 2              | 10.00                   | 0.00                         |
| 6   | 95               | 104           | -9.00                       | 3              | -10.78                  | 1.78                         |
| 7   | 102              | 103           | -1.00                       | 1              | 0.17                    | -1.17                        |
| 8   | 112              | 102           | 10.00                       | 2              | 10.00                   | 0.00                         |
| 9   | 92               | 103.67        | -11.67                      | 3              | -10.78                  | -0.89                        |
| 10  | 107              | 107           | 0.00                        | 1              | 0.17                    | -0.17                        |

---

### Explanation Recap

1. **Trend \(T_t\):**  
   Calculated using a centered moving average (here with window \(p=3\)) to smooth the series.

2. **Detrended Series \(D_t\):**  
   Obtained by subtracting the trend from the observed data: \(D_t = Y_t - T_t\).

3. **Seasonal Component \(S_t\):**  
   For each position in the seasonal cycle (here positions 1, 2, and 3), average the detrended values. These averages represent the typical seasonal effect for that cycle position.

4. **Residual \(R_t\):**  
   The remaining variation is computed as \(R_t = D_t - S_t\).

This table illustrates the complete process—from the observed data through trend estimation, detrending, seasonal factor calculation, and finally isolating the residual component. This breakdown is the mathematical foundation for seasonal decomposition, allowing you to understand the separate effects contributing to the time series.