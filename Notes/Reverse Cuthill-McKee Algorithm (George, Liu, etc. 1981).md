# Reverse Cuthill-McKee Algorithm [(George, Liu, etc. 1981)](http://heath.cs.illinois.edu/courses/cs598mh/george_liu.pdf) 



## Terminology
Let A be an $n\times n$ symmetric positive matrixm with entries $a_{ij}$
- $f_i(A)$ denotes the column substrcipt of the first non-zero component in row $i$ of A:
$$
f_i(A) = min\{j|a_{ij}\neq 0\}
$$
- $\beta_i(A)$ denotes the **i-th bandwidth** of A:
$$
\beta_i(A) = i -f_i(A)
$$
- $\beta(A)$ denotes the **bandwith** of A:
$$
\beta(A) = \max \{\beta_i(A)|1\le i \le n\}
$$

<img src="https://p.ipic.vip/rdm252.png" alt="Example of $f_i(A) and \beta_i(A)$" style="zoom:60%;" />

- $Env(A)$ denotes the **envelop** of A:
  $$
  Env(A) = \{\{i,j\}|0<i-j\le\beta_i(A\}
  $$

- **Envelop size** = $|Env(A)|$

  <img src="https://p.ipic.vip/77uhoy.png" alt="Illustration of envelope of A" style="zoom:100%;" />

- $w_i(A)$ denotes the **i-th frontwidth** of A, the number of "active" rows at i-th step:
  $$
  w_i(A) = |\{j>i|\{i,j\}\in Env(A)\}|
  $$
  <img src="https://p.ipic.vip/w35gav.png" alt="Illustration of the i-th bandwidth and frontwidth" style="zoom:80%;" />

- **Frontwidth** : $w(A) = \max\{w_i(A)|1\le i\le n \}$

- Define **Eccentricity** $l(x)$ as the maximum distance that root $x$ could reach within the graph:
  $$
  l(x) = max\{d(x,y)|y\in X\}
  $$

- A node $x\in X$ is said to be a **peripheral** nodes if its *eccentricity* is equal to the diameter of the graph.

## Theorem

- The Adjacent set $Adj(\{x_1,\cdots,x_i\})$ shall be referred to as the i-th front of the labelled graph, and its size the i-th frontwidth
  $$
  For\ i<j,\{i,j\}\in Env(A)\ if\ and\ only\ if\ x_j\in Adj(\{x_1,\cdots,x_i\})
  $$
  <img src="https://p.ipic.vip/i3mqrt.png" alt="Adjacent sets" style="zoom:67%;" />

  <img src="https://p.ipic.vip/lif3pd.png" alt="s" style="zoom:67%;" />



## Algorithm

### Discussion 

- **How the bandwidth and envelop of the matrix affects the fill-in ?**

  - When a system of linear equations has a band matrix of coefficients and the system is solved by Guassian elimination, with pivots taken from the diagonal, **all arithmetic is confined to band and no new zero elements are generated outside of the band**.

  - **To minimize the bandwidth of the row associated with z, node z should be ordered as soon as possible after y.**

  - **Greedy policy**: The Cuthill-McKee scheme can be regarded as a method that reduces the bandwidth of a matrix via a local minimization of the $\beta_i$'s. **This suggests that the scheme can be used as a method to reduce the profile/envelope** 

    ![bandwidth](https://p.ipic.vip/6zuhh7.png)

- Why reverse ?

  - Reversing the Cuthill-McKee ordering ofren turns out to be much superior to the original ordering in terms of profile reduction, although tha bandwidth remains unchanged.

- **Heuristic searching for peripheral node**
  - Peripheral node is the ideal starting point in RCM and many other algorithms, while it is expensive to find with the time complexity bound of $O(|X|^2)$. Instead we search the pseudo-peripheral node by iterating in the level strcture of the graph.



### Implementation

**RCM algorithm**

<img src="https://p.ipic.vip/9ifpni.png" alt="RCM algorithm main function" style="zoom:67%;" />

**Finding pseudo-peripheral nodes**

<img src="https://p.ipic.vip/qcvh2p.png" alt="Finding pseudo-peripheral nodes" style="zoom:80%;" />

