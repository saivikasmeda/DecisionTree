from DecisionTree import *
import pandas as pd
from sklearn import model_selection

# Car Evaluation Data Set   - 0.96 154-152 0.97 53-52
# header = ['buying', 'maint', 'doors', 'persons', 'lug_boot','safety','Class']

# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot','safety','Class'])



# Annealing Data Set - 0.97 30-28 0.98 9-8
header = ['family', 'product', 'steel', 'carbon', 'hardness','temper_rolling','condition','formability','strength','non-ageing','surface-finish','surface-quality', 'enamelability', 'bc','bf','bt','bw/me','bl','m','chrom','phos','cbond','marvi','exptl','ferro','corr','blue/bright/varn/clean','lustre','jurofm','s','p','shape','thick','width','len','oil','bore','packing','class']

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data', header=None, names=['family', 'product', 'steel', 'carbon', 'hardness','temper_rolling','condition','formability','strength','non-ageing','surface-finish','surface-quality', 'enamelability', 'bc','bf','bt','bw/me','bl','m','chrom','phos','cbond','marvi','exptl','ferro','corr','blue/bright/varn/clean','lustre','jurofm','s','p','shape','thick','width','len','oil','bore','packing','class'])


#  Iris data set - 13 -11 0.95 ; 4 -3
# header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])

lst = df.values.tolist()
t = build_tree(lst, header)
# print_tree(t)

# print("********** Leaf nodes ****************")
# leaves = getLeafNodes(t)
# for leaf in leaves:
#     print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
# print("********** Non-leaf nodes ****************")
# innerNodes = getInnerNodes(t)
# for inner in innerNodes:
#     print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
result_tree  = build_tree(train,header)   

print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("------------------------------------------")
print("Accuracy on test = " + str(acc))
print("\n\n")
print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
print("No.of leaves nodes",len(leaves))
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
print("No.of Inner Nodes ",len(innerNodes))
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))


## TODO: You have to decide on a pruning strategy
# inner nodes
temp_acc = acc
innerNodes.sort(key=lambda x: x.id, reverse=True)
pruning_List = []
max_pruning_List=[]
temp = 0.0
for i in range(len(innerNodes)):
    pruning_List.append(innerNodes[i].id)
    t_pruned = prune_tree(t, pruning_List)
    acc_test = computeAccuracy(test, t)
    if (temp <= acc_test):
        max_pruning_List = pruning_List[:]
        temp = acc_test
        
    if acc_test <temp_acc:
        break
    

result_prune_tree = prune_tree(result_tree,max_pruning_List)


print("*************Tree after pruning*******")
print_tree(result_prune_tree )
result_acc = computeAccuracy(test,result_prune_tree)
print("------------------------------------------")
print("Accuracy on test after pruning = " + str(result_acc))


print("\n\n")


print("********** Leaf nodes after pruning ****************")
leaves_result = getLeafNodes(result_prune_tree,[])
print("No. of leaves nodes after pruning",len(leaves_result))
for leaf in leaves_result:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes after pruning ****************")
innerNodes_result = getInnerNodes(result_prune_tree,[])
print("No. of Inner Nodes after pruning ",len(innerNodes_result))
for inner in innerNodes_result:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
    
    
    
    
