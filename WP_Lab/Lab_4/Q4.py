class ParenthesesValidator:
    def is_valid(self, s):
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        for char in s:
            if char in mapping:
                top_element = stack.pop() if stack else '#'
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
        return not stack

# Example usage:
validator = ParenthesesValidator()
print(validator.is_valid("()[]{}"))  # True
print(validator.is_valid("([)]"))    # False
