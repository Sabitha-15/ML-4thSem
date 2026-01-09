def read_matrix(rows, cols):
    matrix = []
    for _ in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)
    return matrix

def transpose(matrix1):
    result_transpose=[[0 for _ in range(len(matrix1))] for _ in range(len(matrix1[0]))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1)):
           result_transpose[j][i]=matrix1[i][j] 
    return result_transpose

row=int(input("enter no of rows of the matrix: "))
col=int(input("enter no of cols of matrix: "))
mat=read_matrix(row,col)
res=transpose(mat)
print(res)
    