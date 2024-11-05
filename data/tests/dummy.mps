NAME           dummy
ROWS
 N  obj     
 L  c1      
 G  c2      
 E  c3          
COLUMNS
    x1        obj                  1   c3                   1
    x1        c2                  -3
    x2        obj                  5   c3                   1
    x2        c1                   1
    x3        obj                  1   c2                   1
    x3        c1                  -1
RHS
    rhs       c1                0   c2                0
    rhs       c3                1
BOUNDS
 UP bnd       x1                  1
 UP bnd       x2                  1
 UP bnd       x3                  6
 LO bnd       x1                  0
 LO bnd       x2                  0
 LO bnd       x3                  -5
ENDATA
