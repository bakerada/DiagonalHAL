Adam Baker


### Abstract
The purpose of my implementation is to exploit known properties of a given matrix, **A** of size  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{R}^{ndxnd}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{R}^{ndxnd}" title="\mathbb{R}^{ndxnd}" /></a>, and perform efficient matrix operations.  The solution enables operations of **A** on large values of *n* and *d* by first creating a compact represenation for **A**.  I have made updates to common algorithms for matrix operations, such as matrix multiplication, LU Decomposition, inverse, and solving linear equations with the compact representation.  Pairing both a new representation and updates to existing algorithms enables operations on **A** not typically feasible on a single device, while reducing the number of calculations necessary complete the operations.  The python library Numpy is the only requriement for the implementation and it can easily be updated to utilize parallel frameworks and disk operations to further increase the speed of the operations and size of **A** respectively.  


