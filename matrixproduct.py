def read_matrix(rows, cols):
    matrix = []
    for _ in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)
    return matrix


def multiply_matrices(matrix_a, matrix_b, rows_a, cols_a, rows_b, cols_b):

    #Multiplies two matrices if dimensions are compatible.
    
    if cols_a != rows_b:
        return None

    # Initialize result matrix with zeros
    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    # Matrix multiplication logic
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(rows_b):
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result_matrix

rows_a = int(input("Enter number of rows of A: "))
cols_a = int(input("Enter number of columns of A: "))
print("Enter matrix A:")
matrix_a = read_matrix(rows_a, cols_a)

rows_b = int(input("Enter number of rows of B: "))
cols_b = int(input("Enter number of columns of B: "))
print("Enter matrix B:")
matrix_b = read_matrix(rows_b, cols_b)

result = multiply_matrices(matrix_a, matrix_b, rows_a, cols_a, rows_b, cols_b)

if result is None:
    print("Matrices are not multiplicable")
else:
    print("Result matrix:")
    for row in result:
        print(row)
