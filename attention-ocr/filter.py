# import glob

# cs2 = set()
# for file in glob.glob('../name-train/*.txt'):
#     content = open(file).read().replace(" ","").lower()
#     if 'f' in content:
#             print(file)
#     cs2 = cs2.union(set(content))

# cs = set()
# import pdb
# content= open('datasets/data/names/charset_size=95.txt').read().strip()
# for line in content.split('\n'):
#     cs.add(line.split('\t')[1])

# # content= open('datasets/data/names2/charset_size=91.txt').read().strip()
# # for line in content.split('\n'):
# #     cs2.append(line.split('\t')[1])


# bt = set()
# for i in (cs2-cs):
#     bt.add(i)

# print(bt)


# import glob

# for file in glob.glob('../val-name/*.txt'):
#     content = open(file).read()
#     if bt.intersection(set(content)).__len__() > 0:
#         print(file)
