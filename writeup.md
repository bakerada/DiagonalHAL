

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
<p align="center" font> 
  An example of matrix <strong>A</strong>, <strong>D</strong><sub>ij</sub> is a diagonal matrix of size <a href="https://www.codecogs.com/eqnedit.php?latex=d&space;x&space;d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d&space;x&space;d" title="d x d" /></a>
</p>  
<br><br>

Given the *n* and *d* properties of matrix **A**, we can now derive new properties of the matrix.  A diagonal matrix is a matrix where the only populated cells occur where the row index equals the column index, **D**<sub>jj</sub>.  With only the diagonal elements of the matrix populated, diagonal matricies become increasingly sparse as the dimension of the matrix increases. For matrix **A**, *d* controls the rate at which the sparsity increases.  Because **A** is made up of non overlapping diagonal matricies, the populated cells of the matrix are known simply by knowing *n* and *d*.


<br><br>
### Dense Representation
The memory requirements of **A** can be significantly reduced by eliminating the non-diagonal elements of each of the <a href="https://www.codecogs.com/eqnedit.php?latex=n^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^2" title="n^2" /></a> diagonal matricies.  For any values of *n* and *d*, matrix **A** contains <a href="https://www.codecogs.com/eqnedit.php?latex=n^2\ast&space;d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^2\ast&space;d" title="n^2\ast d" /></a> diagonal elements.  The dense represention reduces memory requirements by a factor of *d*.  


  

