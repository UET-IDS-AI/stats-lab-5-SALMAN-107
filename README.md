[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cZasLbJg)
# Statistics Lab – Continuous Random Variables

This lab explores probability theory concepts used in data-science:

• CDF and PDF  
• Continuous distributions  
• Exponential distribution  
• Gaussian distribution  
• Conditional probability  
• Simulation and numerical verification  

You will implement **two problems**.

The goal is to connect **analytical probability formulas** with **simulation-based estimation**.

---

# Question 1 — CDF, PDF, and Probabilities (Exponential Distribution)

A random variable has CDF:

F_X(x) = (1 − e^{-λx})u(x)

where  
λ > 0 and u(x) is the unit step function.

### Tasks

1. Derive the **PDF from the CDF**.
2. Write a function that returns the PDF.
3. Write a function that computes

P(a < X < b)

using the PDF integral.
4. Write a function that **simulates exponential samples** and estimates the same probability.

### Expected Result

For λ = 1, a = 2, b = 5:

P(2 < X < 5) = e^{-2} − e^{-5}

---

# Question 2 — Bayesian Classification (Gaussian Model)

Two groups of swimmers compete:

Group A (fast swimmers)  
Group B (slow swimmers)

Their finishing time distributions are:

Group A  
X ~ N(40, 4)

Group B  
X ~ N(45, 4)

Prior probabilities

P(A) = 0.3  
P(B) = 0.7

A swimmer finishes in **42 seconds**.

### Tasks

1. Implement a **Gaussian PDF function**.
2. Compute the likelihood

f_{X|A}(42), f_{X|B}(42)

3. Use **Bayes rule** to compute

P(B | X = 42)

4. Write a function that **simulates swimmers** and estimates the same posterior probability.

---

