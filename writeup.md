

### Abstract
The purpose of my implementation is to exploit known properties of a given matrix, **A** of size  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{R}^{ndxnd}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{R}^{ndxnd}" title="\mathbb{R}^{ndxnd}" /></a>, and perform efficient matrix operations.  The solution enables operations of **A** on large values of *n* and *d* by first creating a compact represenation for **A**.  I have made updates to common algorithms for matrix operations, such as matrix multiplication, LU Decomposition, inverse, and solving linear equations with the compact representation.  Pairing both a new representation and updates to existing algorithms enables operations on **A** not typically feasible on a single device, while reducing the number of calculations necessary complete the operations.  The python library Numpy is the only requriement for the implementation and it can easily be updated to utilize parallel frameworks and disk operations to further increase the speed of the operations and size of **A** respectively.  

### Background

There are two important characteristics of matrix **A**, which enable compact and efficient operations:

* **A** is of size  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{R}^{ndxnd}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{R}^{ndxnd}" title="\mathbb{R}^{ndxnd}" /></a>
* **A** are <a href="https://www.codecogs.com/eqnedit.php?latex=n^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^2" title="n^2" /></a> non overlapping diagonal matricies of size <a href="https://www.codecogs.com/eqnedit.php?latex=d&space;x&space;d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d&space;x&space;d" title="d x d" /></a>
<br><br>
<p align="center"> 
  <a href="https://www.codecogs.com/eqnedit.php?  latex=\begin{bmatrix}&space;D_{11}&space;&&space;D_{12}&space;&&space;...&space;&&space;D_{1n}\\&space;D_{21}&&space;D_{22}&space;&&space;...&space;&&space;D_{2n}\\&space;D_{31}&&space;D_{32}&space;&&space;...&space;&&space;D_{3n}&space;\\&space;D_{41}&&space;D_{42}&space;&&space;...&space;&&space;D_{4n}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;D_{11}&space;&&space;D_{12}&space;&&space;...&space;&&space;D_{1n}\\&space;D_{21}&&space;D_{22}&space;&&space;...&space;&&space;D_{2n}\\&space;D_{31}&&space;D_{32}&space;&&space;...&space;&&space;D_{3n}&space;\\&space;D_{41}&&space;D_{42}&space;&&space;...&space;&&space;D_{4n}&space;\end{bmatrix}" title="\begin{bmatrix} D_{11} & D_{12} & ... & D_{1n}\\ D_{21}& D_{22} & ... & D_{2n}\\ D_{31}& D_{32} & ... & D_{3n} \\ D_{41}& D_{42} & ... & D_{4n} \end{bmatrix}" /></a>
  
</p>
<p align="center"> 
  An example of matrix <strong>A</strong>, <strong>D</strong><sub>ij</sub> is a diagonal matrix of size <a href="https://www.codecogs.com/eqnedit.php?latex=d&space;x&space;d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d&space;x&space;d" title="d x d" /></a>
</p>  
<br><br>

Given the *n* and *d* properties of matrix **A**, we can now derive new properties of the matrix.  A diagonal matrix is a matrix where the only populated cells occur where the row index equals the column index, **D**<sub>jj</sub>.  With only the diagonal elements of the matrix populated, diagonal matricies become increasingly sparse as the dimension of the matrix increases. For matrix **A**, *d* controls the rate at which the sparsity increases.  Because **A** is made up of non overlapping diagonal matricies, the populated cells of the matrix are known simply by knowing *n* and *d*.


<br><br>
### Dense Representation
The memory requirements of **A** can be significantly reduced by eliminating the non-diagonal elements of each of the <a href="https://www.codecogs.com/eqnedit.php?latex=n^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^2" title="n^2" /></a> diagonal matricies.  For any values of *n* and *d*, matrix **A** contains <a href="https://www.codecogs.com/eqnedit.php?latex=n^2\ast&space;d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^2\ast&space;d" title="n^2\ast d" /></a> diagonal elements.  The dense represention reduces memory requirements by a factor of *d*.

<br><br>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;1&space;&&space;0&space;&&space;1&space;&&space;0&space;\\&space;0&space;&&space;1&space;&&space;0&space;&1&space;\\&space;1&space;&&space;0&space;&&space;1&space;&&space;0\\&space;0&space;&&space;1&space;&&space;0&space;&1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;1&space;&&space;0&space;&&space;1&space;&&space;0&space;\\&space;0&space;&&space;1&space;&&space;0&space;&1&space;\\&space;1&space;&&space;0&space;&&space;1&space;&&space;0\\&space;0&space;&&space;1&space;&&space;0&space;&1&space;\end{bmatrix}" title="\begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 &1 \\ 1 & 0 & 1 & 0\\ 0 & 1 & 0 &1 \end{bmatrix}" /></a>&nbsp&nbsp&nbsp&nbsp&nbsp <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;1&space;&1&space;\\&space;1&space;&1&space;\\&space;1&space;&1&space;\\&space;1&space;&1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;1&space;&1&space;\\&space;1&space;&1&space;\\&space;1&space;&1&space;\\&space;1&space;&1&space;\end{bmatrix}" title="\begin{bmatrix} 1 &1 \\ 1 &1 \\ 1 &1 \\ 1 &1 \end{bmatrix}" /></a>
</p>

<p align="center">
  Figure 1: The left matrix is a valid example of <strong>A</strong>, with n=2 and d=2.  The right matrix is an example of **A** deconstructed into a dense representation.
</p>

<br>

The deconstruction of **A** into a dense represenation is trival, but to perform operations, such as matrix multiplication with another matrix, additional information is necessary.  Each dense representation of **A** is accompanied by another vector(s), which will be referred to as the basis vector(s) throughout this write up.  The basis vectors contain the indicies of the non-diagonal elements of **A** before is deconstructed into the sparse representation.  The basis vectors do not contain every unique index of the non-diagonal elements of **A**.  Rather, there are *d* basis vectors of size *n*, where each vector represents a column and each element of the vector represents the non diagonal elements in the rows of **A**.  The basis vectors are representative of each non-overlapping batch of *d* columns in **A**.
<br>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;\begin{bmatrix}&space;0&2\\&space;\end{bmatrix}\\&space;\begin{bmatrix}&space;1&3\\&space;\end{bmatrix}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;\begin{bmatrix}&space;0&2\\&space;\end{bmatrix}\\&space;\begin{bmatrix}&space;1&3\\&space;\end{bmatrix}&space;\end{bmatrix}" title="\begin{bmatrix} \begin{bmatrix} 0&2\\ \end{bmatrix}\\ \begin{bmatrix} 1&3\\ \end{bmatrix} \end{bmatrix}" /></a>
</p>
<p align="center">
  Figure 2. An example set of basis vectors from the matrix provided in figure 1.  For column 0, rows 0 and 2 are populated, while for column 1, rows 1 and 3 are populated.
</p>

The basis vectors enables the implementation to perform the necessary matrix operations, while requiring a minimum amount of memory.  The basis include the popualted cells for each column, however it can be further reduced by only storing the populated rows for each *d* block, rather than all rows.  If memory is an issue, it is a trivial update, however the basis vectors are frequently leveraged throughout the implementation and make for straight forward indexing.  

<br>

### Matrix Multiplication

A core matrix operation is matrix multiplication, where a single matrix is formed by the compsition of two discrete matricies.  If **B** is a *m*x*m* matrix and **C** is a *m*x*n* matrix, matrix multiplication is the summation of the products of *m* row values in **B** by the *m* column values of **C** for each row-column pairing.  The result of matrix multiplication a *m*x*n* matrix by a *m*x*n* matrix is a matrix of size *m*x*n*.

<p align='center'>
  <a href="https://www.codecogs.com/eqnedit.php?latex=B=\begin{bmatrix}&space;1&space;&3&space;\\&space;2&space;&4&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B=\begin{bmatrix}&space;1&space;&3&space;\\&space;2&space;&4&space;\end{bmatrix}" title="B=\begin{bmatrix} 1 &3 \\ 2 &4 \end{bmatrix}" /></a>&nbsp&nbsp
  <a href="https://www.codecogs.com/eqnedit.php?latex=C=\begin{bmatrix}&space;3&space;&5&space;\\&space;1&space;&2&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C=\begin{bmatrix}&space;3&space;&5&space;\\&space;1&space;&2&space;\end{bmatrix}" title="C=\begin{bmatrix} 3 &5 \\ 1 &2 \end{bmatrix}" /></a>
</p>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=BC=\begin{bmatrix}&space;1*3&space;&plus;&space;3*1&space;&1*5&plus;3*2&space;\\&space;2*3&space;&plus;4*1&space;&2&space;*5&plus;4*2&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?BC=\begin{bmatrix}&space;1*3&space;&plus;&space;3*1&space;&1*5&plus;3*2&space;\\&space;2*3&space;&plus;4*1&space;&2&space;*5&plus;4*2&space;\end{bmatrix}" title="BC=\begin{bmatrix} 1*3 + 3*1 &1*5+3*2 \\ 2*3 +4*1 &2 *5+4*2 \end{bmatrix}" /></a>
</p>
<p align="center">
  Figure 3. A sample matrix multiplication between two matricies **B** and **C**
</p>

If a value is zero within the matrix, it does not contribute to the composition of the two matricies during matrix multiplication.  Therefore, to perform matrix multiplication between **A** and another matrix,**X**, only the cells within **X** which correspond to a non-diagonal element within **A** are necessary for calculating matrix multiplication.  To implement matrix multiplication, the rows of **X** are indexed utilizing the basis vectors to create a compact representation of the columns of **X**.  If **X** is a *m*x*k* matrix, the compact representation of **X** is a matrix of size *d*x*n*x*k* of only the necessary cells from **X**.  Once indexing of **X** is complete, to complete the matrix multiplication operation the dot product between each row in the dense representation of **A** and the corresponding *k*x*n* matrix.  To choose the proper *k*x*n* matrix, the remainder of the row index of **A** by *d* indicates which index to select within the *d*x*n*x*k* matrix.  
<br>

<p align="center">
  <img src="https://github.com/bakerada/DiagonalHAL/blob/master/numpyvsmine.png" alt="Numpy vs Implementation">
  <br>Figure 4: A comparison of matrix multiplication using numpy versus the dense representation <br>
</p>

The matrix multiplication implementation allows for significantly larger values of *n* and *d*.  For smaller values on *n* and *d*, the sparse implemenation of matrix multiplication has runtimes near that of base numpy.  However, as *n* and *d* increase, the custom implementation becomes significantly faster, as shown in figure 4.  While the increase speed of the matrix multiplication operation is beneficial, the true value of the implemenation is the decrease in memory consumption.  To perform a matrix multiplication between **A** with *n*=1000 and *d*=1000 and **X** of size 1000x1000, 26GB are utilized with the new implementation. With base numpy, more than 32GB are required when *n*=300 and *d*=300.

<br>

### LU Decomposition

The other availible operations in the implementation include finding the inverse and determinant of **A**, as well as solving linear systems of equations with **A**.  To efficiently perform these operations, the implementation leverages LU decomposition.  LU decompostion is the process of breaking down a square matrix into two components, a unit lower triangular matrix **L** and an upper triangular matrix **U**.  The product of **L** and **U** equals the original matrix [1].
<br>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=LU=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LU=A" title="LU=A" /></a><br>LU Decomposition
</p>

<br>

LU decomposition is chosen as the engine for majority of the matrix operations primarily due to its applicability and number of operations can be derived from **L** and **U**, but it also maitains the same sparsity patterns as **A**.  Therefore, non diagonal elements in **A** will be zero in both **U** and **L** matricies.  Because sparsity is maintained, the basis vectors utilized in the matrix multiplication operations can also be leveraged to perform LU decomposition on the dense representation of **A**.  The Doolittle algorithm with pivoting is the method for implementing LU decomposition for the dense representation of **A**.  To solve for **LU** with Doolittle, an iterative process through each column in **A** solving for first the **U** values in the current column, then solving for the values in the column of **L**.  The calculation for each value of **U** and **L** can be seen in figure 5.

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=u_{ij}=&space;a_{ij}&space;-&space;\sum^{i-1}&space;_{k=1}&space;u_{kj}l_{ik}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{ij}=&space;a_{ij}&space;-&space;\sum^{i-1}&space;_{k=1}&space;u_{kj}l_{ik}" title="u_{ij}= a_{ij} - \sum^{i-1} _{k=1} u_{kj}l_{ik}" /></a>
</p>

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=l_{ij}=&space;(a_{ij}&space;-&space;\sum^{j-1}&space;_{k=1}&space;u_{kj}l_{ik})/(u_{jj})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{ij}=&space;(a_{ij}&space;-&space;\sum^{j-1}&space;_{k=1}&space;u_{kj}l_{ik})/(u_{jj})" title="l_{ij}= (a_{ij} - \sum^{j-1} _{k=1} u_{kj}l_{ik})/(u_{jj})" /></a><br> Figure 5: j is the current column,i is the current value in j, while k represents the values up to i
</p>

The formulats in figure 6, nested within a loop through all columns in **A**, total up to numerous iterations and calculations.  Since sparsity of **A** is represented in both **U** and **L**, the implementation utilizes the basis vectors to only perform **U** and **L** calculations on the necessary matricies.  The implementation pivots the dense representation directly, then performs the **U** and **L** calculations on the dense represenation of **A**.  By performing LU decomposition on the dense representation we can eliminate many of the iterations of the standard Doolittle algorithm.  For each column in **A**, only an single for-loop of *n* values is necessary to calculate the dense **LU** matricies, reducing the iterations by a factor over *d*.  The implementation also saves on memory, as the created **L** and **U** matricies follow the same dense representation as **A**.

<br>
### Solving Linear Equations
A key


### References

[1] Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T. "LU Decomposition and Its Applications." §2.3 in Numerical Recipes in FORTRAN: The Art of Scientific Computing, 2nd ed. Cambridge, England: Cambridge University Press, pp. 34-42, 1992.
[2] Timothy Sauer. 2011. Numerical Analysis (2nd ed.). Addison-Wesley Publishing Company, ch 2, USA.



