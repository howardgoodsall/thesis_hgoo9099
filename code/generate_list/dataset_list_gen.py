"""
Generates lists of images for input to the rest of the code. Adapted from https://github.com/tim-learn/Generate_list. 
Usage: python3 dataset_list_gen.py <scale factor between 0 and 1 - e.g. 0.5>
"""
import os
import sys

if(len(sys.argv) != 2):
	scale_factor = 1.0
else:
	scale_factor = float(sys.argv[1])

if(scale_factor >1 or scale_factor<0):
	print("usage: python3 dataset_list_gen.py <scale factor between 0 and 1 - e.g. 0.5>")
	sys.exit(0)

#Replace with path to location of dataset
# ---
# ---
folder = r"C:\Users\Howard\Desktop\Honours\thesis_hgoo9099\code\datasets\office"
# ---
# ---

if('/' in folder):
	dataset_name = folder.split("/")[-1]
elif('\\' in folder):
	dataset_name = folder.split('\\')[-1]	
else:
	print("Could not access folder at path: " + folder)
	sys.exit(0)

print(dataset_name)
domains = os.listdir(folder)
domains.sort()

new_dir = dataset_name + "_" + str(scale_factor)
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
    
os.chdir(new_dir)

for d in range(len(domains)):
	dom = domains[d]
	if os.path.isdir(os.path.join(folder, dom)):
		dom_new = dom.replace(" ","_")
		#os.rename(os.path.join(folder, dom), os.path.join(folder, dom_new))

		classes = os.listdir(os.path.join(folder, dom_new))
		classes.sort()
		
		f = open(dom_new + "_list.txt", "w")
		f_train = open(dom_new + "_train_list.txt", "w")
		f_test = open(dom_new + "_test_list.txt", "w")
		for c in range(len(classes)):
			cla = classes[c]
			cla_new = cla.replace(" ","_")
			#os.rename(os.path.join(folder, dom_new, cla), os.path.join(folder, dom_new, cla_new))
			files = os.listdir(os.path.join(folder, dom_new, cla_new))
			train_files = files[-(int(len(files)*0.9)+1):]#Default train test split is 9:1
			test_files = files[:int(len(files)*0.1)]
			train_files = train_files[:int(len(train_files)*scale_factor)]#Further reduce train files by scale factor
			print(cla_new + ": Total: " + str(len(files)) + " Train: " + str(len(train_files)) + " Test: " + str(len(test_files)))
			files.sort()#order that files are read by os is random
			#files = files
			for file in files:
				file_new = file.replace(" ","_")
				os.rename(os.path.join(folder, dom_new, cla_new, file), os.path.join(folder, dom_new, cla_new, file_new))
				f.write('{:} {:}\n'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
			for file in train_files:
				file_new = file.replace(" ","_")
				os.rename(os.path.join(folder, dom_new, cla_new, file), os.path.join(folder, dom_new, cla_new, file_new))
				f_train.write('{:} {:}\n'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
			for file in test_files:
				file_new = file.replace(" ","_")
				os.rename(os.path.join(folder, dom_new, cla_new, file), os.path.join(folder, dom_new, cla_new, file_new))
				f_test.write('{:} {:}\n'.format(os.path.join(folder, dom_new, cla_new, file_new), c))
		f.close()
		f_train.close()
		f_test.close()
		