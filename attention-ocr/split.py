import glob
import os
fs = glob.glob("number/*.jpg")
n_train = int(len(fs)-200)
n_val = 200
trainfiles = fs[:n_train]
valfiles = fs[n_train:]
#testfiles = fs[n_train+n_val:]

if not os.path.isdir('dateback-train'):
    os.makedirs("dateback-train")
    os.makedirs("dateback-val")
#os.makedirs("address-test")

def move(files, dirr):
	# import pdb; pdb.set_trace()
	for f in files:
		try:
			name = f.replace("number/","").replace(".jpg", "")
			os.rename("number/"+name+'.jpg', dirr+"/"+name+".jpg" )
			os.rename("number/"+name+'.txt', dirr+"/"+name+".txt" )
		except Exception as err:
			print(err)

move(trainfiles, "dateback-train")
move(valfiles, "dateback-val")
#move(testfiles, "address-test")


#import os
#files = set()
#for f in os.listdir("address-val"):
#    files.add(f.replace(".jpg", "").replace(".txt", ""))
#for f in os.listdir("address-test"):
#    files.add(f.replace(".jpg", "").replace(".txt", ""))
#
#with open('valtest.txt', 'w+') as fp:
#    fp.write('\n'.join(files))
