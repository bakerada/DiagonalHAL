

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





  

