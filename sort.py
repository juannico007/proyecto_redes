a = [[3,4,5],[1,2,3],[9,8,2]]

print(a)
def bubbleSort(arr):
    
    n = len(arr[1])
 
    # Traverse through all array elements
    for i in range(n-1):
    # range(n) also work but outer loop will
    # repeat one time more than needed.
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[1][j] < arr[1][j + 1] :
                arr[0][j], arr[0][j + 1] = arr[0][j + 1], arr[0][j]
                arr[1][j], arr[1][j + 1] = arr[1][j + 1], arr[1][j]
                arr[2][j], arr[2][j + 1] = arr[2][j + 1], arr[2][j]
    return arr

print(bubbleSort(a))

