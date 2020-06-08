import numpy
from numpy import array
from numpy import mean
from numpy import cov, var
from PIL import Image
from numpy.linalg import eigh, norm
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import math


f = open('ts.txt', 'r')
train_images = []
tf = open('tss.txt', 'r')
test_images = []
line_list1 = f.readlines()
#line_list1.pop()
line_list2 = tf.readlines()
for line in line_list1:
	line = line.split("-")    
	train_images.append(
            ((numpy.asarray(Image.open(line[0]).convert('L').resize((64, 64))).flatten()), line[1])) 

for line in line_list2:
    test_images.append(numpy.asarray(Image.open(line.split('\n')[0]).convert('L').resize((64, 64))).flatten())
    
images = []

for (image, name) in train_images:
	images.append(image)

matrix = numpy.asarray(images)
#print(matrix)

avg = mean(matrix.T, axis=1)
center = matrix - avg
variance = cov(center.T)
values, vectors = eigh(variance)

feat_vec = numpy.flip(vectors)[:,:32]
norm_line = feat_vec.T.dot(center.T)

vec = feat_vec
line = norm_line.T
avg = avg
classed_eigen = dict()

for index, arr in enumerate(line):
		if train_images[index][1] not in classed_eigen:
			classed_eigen[train_images[index][1]] = list()
		classed_eigen[train_images[index][1]].append(arr) 
	
for key in classed_eigen:
		classed_eigen[key] = numpy.asarray(classed_eigen[key])

avgg = {}
vari = {}
for name in classed_eigen:
	arr = classed_eigen[name]
	
	mu = [mean(col) for col in arr.T]
	
	sigma_sq = var(arr.T, axis=1)

	if name not in avgg:
		avgg[name] = 0
		vari[name] = 0
	avgg[name] = mu
	vari[name] = sigma_sq
        
meuu = avgg
sigsq = vari

matr = numpy.asarray(test_images)
#print(matr)

cc = matr - avg

test_norm_line = vec.T.dot(cc.T)
test_line = test_norm_line.T
    
prod = 1
max_val = -9999
max_class = list()
for vec in test_line:
	temp_name = 'X'
	max_val = -9999
	for name in meuu:
		prod = 1	
		for index in range(len(vec)): 
			p_x_1 = (2 * 3.14 * sigsq[name][index]) ** 0.5
			ra = (-(vec[index] - meuu[name][index]) ** 2) / (2*sigsq[name][index])
			p_x_2 = math.exp(ra)
			p_x = p_x_2/p_x_1
			prod *= p_x 
			
		if prod > max_val:
			max_val = prod
			temp_name = name
	max_class.append(temp_name)
names = max_class
#print((len(train_images)/6), ' Images per Class have been used to Train the Model')
#print('Using ', len(test_images), ' Images per Class have been used to Test the Model')
#print('\n Training Data Size: ', len(train_images))
#print('Testing Data Size: ', len(test_images))
droness = list()
fjets = list()
helicopts = list()
missiles = list()
pplanes = list()
rockets = list()
#dronesfound = 0;fjetsfound = 0;helicoptersfound = 0
#missilesfound = 0;pplanesfound = 0;rocketsfound = 0

tnd = 0; fnd = 0; fpd = 0; tpd = 0;
tnf = 0; fnf = 0; fpf = 0; tpf = 0;
tnh = 0; fnh = 0; fph = 0; tph = 0;
tnm = 0; fnm = 0; fpm = 0; tpm = 0;
tnp = 0; fnp = 0; fpp = 0; tpp = 0;
tnr = 0; fnr = 0; fpr = 0; tpr = 0;
for count in range(len(test_images)):
    
    # Check drone images
    if count < (len(test_images)/6):
        if (names[count])[0] == 'd':
            tpd += 1
            tnf += 1
            tnh += 1
            tnr += 1
            tnm += 1
            tnp += 1
        if (names[count])[0] == 'f':
            fnd += 1
            fpf += 1
            tnp += 1
            tnr += 1
            tnh += 1
            tnm += 1
        if (names[count])[0] == 'h':
            fnd += 1
            tnf += 1
            tnr += 1
            tnm += 1
            tnp += 1
            fph += 1
        if (names[count])[0] == 'm':
            fnd += 1
            tnf += 1
            tnr += 1
            tnp += 1
            tnh += 1
            fpm += 1
        if (names[count])[0] == 'p':
            fnd += 1
            tnf += 1
            tnr += 1
            fpp += 1
            tnm += 1
            tnh += 1
        if (names[count])[0] == 'r':
            fnd += 1
            tnf += 1
            tnh += 1
            tnp += 1
            tnm += 1
            fpr += 1
#        else:
#            tnd += 1
            
    # Check fighterjet images        
    if count < (len(test_images)/3) and count >= (len(test_images)/6):
        if (names[count])[0] == 'd':
            fnf += 1
            fpd += 1
            tnm += 1
            tnr += 1
            tnp += 1
            tnh += 1
        if (names[count])[0] == 'f':
            tpf += 1
            tnd += 1
            tnr += 1
            tnp += 1
            tnm += 1
            tnh += 1
        if (names[count])[0] == 'h':
            fnf += 1
            tnp += 1
            tnr += 1
            tnd += 1
            tnm += 1
            fph += 1
        if (names[count])[0] == 'm':
            fnf += 1
            fpm += 1
            tnd += 1
            tnp += 1
            tnr += 1
            tnh += 1
        if (names[count])[0] == 'p':
            fnf += 1
            tnd += 1
            fpp += 1
            tnr += 1
            tnm += 1
            tnh += 1
        if (names[count])[0] == 'r':
            fnf += 1
            tnd += 1
            tnh += 1
            fpr += 1
            tnm += 1
            tnp += 1
#        else:
#            tnf += 1
     
     # Check Helicopter Images
    if count < (len(test_images)/2) and count >= (len(test_images)/3):
        if (names[count])[0] == 'd':
            fnh += 1
            fpd += 1
            tnm += 1
            tnr += 1
            tnp += 1
            tnf += 1
        if (names[count])[0] == 'f':
            fnh += 1
            tnr += 1
            tnd += 1
            tnp += 1
            fpf += 1
            tnm += 1
        if (names[count])[0] == 'h':
            tph += 1
            tnr += 1
            tnd += 1
            tnp += 1
            tnm += 1
            tnf += 1
        if (names[count])[0] == 'm':
            fnh += 1
            tnd += 1
            tnr += 1
            tnp += 1
            fpm += 1
            tnf += 1
        if (names[count])[0] == 'p':
            fnh += 1
            fpp += 1
            tnm += 1
            tnr += 1
            tnd += 1
            tnf += 1
        if (names[count])[0] == 'r':
            fnh += 1
            tnd += 1
            tnp += 1
            fpr += 1
            tnm += 1
            tnf += 1
#        else:
#            tnh += 1   
            
    # Check missile images
    if count < (len(test_images)/(6/4)) and count >= (len(test_images)/2):
        if (names[count])[0] == 'd':
            fnm += 1
            fpd += 1
            tnf += 1
            tnr += 1
            tnh += 1
            tnp += 1
        if (names[count])[0] == 'f':
            fnm += 1
            tnp += 1 
            tnr += 1
            tnd += 1
            fpf += 1
            tnh += 1
        if (names[count])[0] == 'h':
            fnm += 1
            tnd += 1
            tnf += 1
            tnr += 1
            tnp += 1
            fph += 1
        if (names[count])[0] == 'm':
            tpm += 1
            tnp += 1
            tnr += 1
            tnd += 1
            tnf += 1
            tnh += 1
        if (names[count])[0] == 'p':
            fnm += 1
            tnd += 1
            tnr += 1
            fpp += 1
            tnf += 1
            tnh += 1
        if (names[count])[0] == 'r':
            fnm += 1
            tnd += 1
            tnp += 1
            fpr += 1
            tnf += 1
            tnh += 1
#        else:
#            tnm += 1
            
    # Check passengerplane images        
    if count < (len(test_images)/(6/5)) and count >= (len(test_images)/(6/4)):
        if (names[count])[0] == 'd':
            fnp += 1
            fpd += 1
            tnf += 1
            tnm += 1
            tnr += 1
            tnh += 1
        if (names[count])[0] == 'f':
            fnp += 1
            tnd += 1
            tnr += 1
            tnm += 1
            fpf += 1
            tnh += 1
        if (names[count])[0] == 'h':
            fnp += 1
            tnd += 1 
            tnr += 1
            tnm += 1
            tnf += 1
            fph += 1
        if (names[count])[0] == 'm':
            fnp += 1
            tnd += 1
            tnh += 1
            tnr += 1
            fpm += 1
            tnf += 1
        if (names[count])[0] == 'p':
            tpp += 1
            tnd += 1
            tnm += 1
            tnf += 1
            tnr += 1
            tnh += 1
        if (names[count])[0] == 'r':
            fnp += 1
            tnd += 1
            tnh += 1
            fpr += 1
            tnm += 1
            tnf += 1
#        else:
#            tnp += 1
            
    # Check rocket images        
    if count < (len(test_images)) and count >= (len(test_images)/(6/5)):
        if (names[count])[0] == 'd':
            fnr += 1
            fpd += 1
            tnh += 1
            tnm += 1
            tnp += 1
            tnf += 1
        if (names[count])[0] == 'f':
            fnr += 1
            tnd += 1
            tnh += 1
            tnp += 1
            fpf += 1
            tnm += 1
        if (names[count])[0] == 'h':
            fnr += 1
            tnd += 1
            tnf += 1
            tnp += 1
            tnm += 1
            fph += 1
        if (names[count])[0] == 'm':
            fnr += 1
            tnd += 1
            tnp += 1
            fpm += 1
            tnh += 1
            tnf += 1
        if (names[count])[0] == 'p':
            fnr += 1
            tnd += 1
            tnf += 1
            fpp += 1
            tnm += 1
            tnh += 1
        if (names[count])[0] == 'r':
            tpr += 1
            tnd += 1
            tnm += 1
            tnp += 1
            tnh += 1
            tnf += 1
#        else:
#            tnr += 1        
    #print('' ,(count+1), 'is a ', names[count])

print('\n            Confusion Matrix for Drones                                     Confusion Matrix for FighterJets\n')
print('        TN  :    ',tnd,'               FP  :     ',fpd,'               TN  :    ',tnf,'               FP  :     ',fpf)
print('        FN  :    ',fnd,'                TP  :     ',tpd,'               FN  :    ',fnf,'                TP  :     ',tpf)
print('\n')    

print('\n            Confusion Matrix for Helicopters                                Confusion Matrix for Missiles\n')
print('        TN  :    ',tnh,'               FP  :     ',fph,'               TN  :    ',tnm,'               FP  :     ',fpm)
print('        FN  :    ',fnh,'                TP  :     ',tph,'               FN  :    ',fnm,'                TP  :     ',tpm)
print('\n')    

print('\n            Confusion Matrix for PassengerPlanes                            Confusion Matrix for Rockets\n')
print('        TN  :    ',tnp,'               FP  :     ',fpp,'               TN  :    ',tnr,'               FP  :     ',fpr)
print('        FN  :    ',fnp,'                TP  :     ',tpp,'               FN  :    ',fnr,'                TP  :     ',tpr)
print('\n')    

