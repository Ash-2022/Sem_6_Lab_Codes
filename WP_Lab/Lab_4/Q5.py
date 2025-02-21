class StringReverser:
    def reverse_words(self, s):
        return ' '.join(reversed(s.split()))

reverser = StringReverser()
print(reverser.reverse_words("Hello World")) 
