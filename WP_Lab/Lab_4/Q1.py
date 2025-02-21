def find_smallest_element(lst):
    if not lst:
        return None
    smallest = lst[0]
    for num in lst:
        if num < smallest:
            smallest = num
    return smallest

# Example usage:
lst = [34, 15, 88, 2, 5]
print("Smallest element:", find_smallest_element(lst))
