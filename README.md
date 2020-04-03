# quadrilateral_measure

<<<<<<< HEAD
find image's specified coloured quadrilateral's 4 precise corners

1 smooth: gaussion smooth with kernel 5x5 is better

2 extract quadrilateral by colour, HSV value will be more stable

3 find a approximative rectangle

4 split 4-edge one by one, and find edge points

5 polygon fit in line: Slope and intercept into K and B

6 fix if K==infinit, do not need Polar coordinates, easy to fix


=== experiences ===

no delite or erode preprocessing,just smooth

from approximative rectangle to precise corner coordinates is efficacious
=======
1 get a picture painted in red and shape is quadrilateral in picture
  the quadrilateral may broken at edge and colour is not smoothly
2 extremely find and draw the 4 lines of quadrilateral
3 return the 4 points in array, for next measure (not implement in this project)

![alt text](https://raw.githubusercontent.com/ligang1841/quadrilateral_measure/master/demo/DSC09672.jpg)

problem
1 not more image test
2 shadow will interference detection
>>>>>>> d1d24c5156b37275fed2c65731e54b176c79075b
