# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:15:23 2017

@author: ahalboabidallah
"""

rows = 13
cols = 11
#range(0,13,5) & range(0,11,5) both return [0, 5, 10]
xBSize = 5
yBSize = 5
for i in range(0, rows, yBSize):
    print('i',i)
    if i+yBSize < rows:
        numRows = yBSize
    else:
        numRows = rows-i
    print('numRows',numRows)
    for j in range(0, cols, xBSize):
        print('j',j)
        if j + xBSize < cols:
            numCols = xBSize
        else:
            numCols = cols-j
        print('numCols',numCols)