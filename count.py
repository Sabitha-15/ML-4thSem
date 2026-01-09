set1={'a','e','i','o','u'}
def vowel_consonant(input_text):
   vowelcount=0
   consonantcount=0
   input_text=input_text.lower()
   for i in range(len(input_text)):
    if input_text[i] in set1:
        vowelcount+=1
    else:
        consonantcount+=1
   return (vowelcount,consonantcount)

text=input("enter the text: ")

#finding vowels and consonants

vowel_consonant=vowel_consonant(text)#a tuple storing vowel count and consonant count
print(vowel_consonant)