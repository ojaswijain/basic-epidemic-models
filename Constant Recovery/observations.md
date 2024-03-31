Major Issues:

- Finding the true value of N
- Keeping k constant seems like a gross oversimplification
- What would the values of beta, gamma be if I = 0?
- Even if k is constant, finding instantaneous values of beta, gamma is difficult because they're not independent of each other

- One dimensional optimization problems for beta, gamma
- Find the optimal beta and k for each day
- SGD on k, finding the optimal beta for each k using the minimization problem:
    err = min(sqrt((beta*S*I+dS/dt)^2+(gamma*I-dR/dt)^2)) over all beta
- argmin(err) = k_optimal

Piecewise median betas:
