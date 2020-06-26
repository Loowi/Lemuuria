dictLabels = {}
finalLabels = {}
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
promoted_to = ['q', 'r', 'b', 'n']

chessNotation = [[letters[i1]+numbers[i2] for i1 in range(8)] for i2 in range(8)]
numberNotation = [[i1+8*i2+1 for i1 in range(8)] for i2 in range(8)]

import itertools

merged1 = list(itertools.chain(*chessNotation))
merged2 = list(itertools.chain(*numberNotation))

for i1 in range(len(merged1)):
        dictLabels[merged2[i1]] = merged1[i1]

allComb = list(itertools.permutations(merged2,2))       
for i in allComb:
    # All moves except promotions
    keyDict = (i[0], i[1], '')
    finalLabels[keyDict] = dictLabels[i[0]] + dictLabels[i[1]]
    
    # Promotions by white
    if i[0] in numberNotation[6][:] and i[1] in numberNotation[7][:]:
        for i2 in promoted_to:
            keyDict = (i[0], i[1], i2)
            finalLabels[keyDict] = dictLabels[i[0]] + dictLabels[i[1]] + i2
        
    # Promotions by black    
    if i[0] in numberNotation[1][:] and i[1] in numberNotation[0][:]:
        for i2 in promoted_to:
            keyDict = (i[0], i[1], i2)
            finalLabels[keyDict] = dictLabels[i[0]] + dictLabels[i[1]] + i2        
        
    