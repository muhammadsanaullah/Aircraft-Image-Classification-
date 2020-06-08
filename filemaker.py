#Creating Training Sample
f = open('ts.txt', 'w')

for i in range(1,1001):
    s = './datasetnb/drone (' + str(i) + ').jpg-drone' + '\n'
    f.write(s)

for i in range(1,1001):
    s = './datasetnb/fighterjet (' + str(i) + ').jpg-fighterjet' + '\n'
    f.write(s)
    
for i in range(1,1001):
    s = './datasetnb/helicopter (' + str(i) + ').jpg-helicopter' + '\n'
    f.write(s)

for i in range(1,1001):
    s = './datasetnb/missile (' + str(i) + ').jpg-missile' + '\n'
    f.write(s)
    
for i in range(1,1001):
    s = './datasetnb/passengerplane (' + str(i) + ').jpg-passengerplane' + '\n'
    f.write(s)

for i in range(1,1001):
    s = './datasetnb/rocket (' + str(i) + ').jpg-rocket' + '\n'
    f.write(s)
f.close()


#Creating Testing Sample
f2 = open('tss.txt','w')
for i in range(1,401):
    s = './datasetnb/ddd (' + str(i) + ').jpg' + '\n'
    f2.write(s)
for i in range(1,401):
    s = './datasetnb/fff (' + str(i) + ').jpg' + '\n'
    f2.write(s)
for i in range(1,401):
    s = './datasetnb/hhh (' + str(i) + ').jpg' + '\n'
    f2.write(s)
for i in range(1,401):
    s = './datasetnb/mmm (' + str(i) + ').jpg' + '\n'
    f2.write(s)
for i in range(1,401):
    s = './datasetnb/ppp (' + str(i) + ').jpg' + '\n'
    f2.write(s)
for i in range(1,401):
    s = './datasetnb/rrr (' + str(i) + ').jpg' + '\n'
    f2.write(s)
f2.close()    
