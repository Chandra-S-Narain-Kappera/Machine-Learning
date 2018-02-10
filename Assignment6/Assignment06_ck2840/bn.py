import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import  NULL
from rpy2.robjects import vectors

# Import bnlearn and grain
bnlearn = rpackages.importr('bnlearn')
grain = rpackages.importr('gRain')

# Reading alarm model and bn.fit model
alarm = bnlearn.read_bif("alarm.bif")

# Plotting the alarm graph
"""
pdf = ro.r('pdf')
pdf("alarm.pdf")
graph = bnlearn.graphviz_plot(alarm)
"""
# Plotting code
grdevices = rpackages.importr('grDevices')
pdf = ro.r('pdf')
pdf("alarm.pdf")
#actual plot
graph = bnlearn.graphviz_plot(alarm)
grdevices.dev_off()

# import utilities
utils = rpackages.importr('utils')

# Read sample 1
sample_1 = utils.read_csv('hw6_q3_data_50k.csv', header = True, sep = ",")

#using hill climbing give DAG, iamb gave partially directed graphs
model1 = bnlearn.hc(sample_1)
grdevices = rpackages.importr('grDevices')
pdf = ro.r('pdf')
pdf("sample_50k.pdf")
#actual plot
graph = bnlearn.graphviz_plot(model1)
grdevices.dev_off()

# Read sample 2

sample_2 = utils.read_csv('hw6_q3_data_500k.csv', header = True, sep = ",")

#using hill climbing give DAG, iamb gave partially directed graphs
model2 = bnlearn.hc(sample_2)
grdevices = rpackages.importr('grDevices')
pdf = ro.r('pdf')
pdf("sample_500k.pdf")
#actual plot
graph = bnlearn.graphviz_plot(model2)
grdevices.dev_off()

# Now we have all the networks lets try tinkering:

# compare score of sample 1 with bnlearn
o_model = bnlearn.bn_net(alarm)
model_1_fit = bnlearn.bn_fit(model1,sample_1)
model_2_fit = bnlearn.bn_fit(model2, sample_2)
#score = bnlearn.score(o_model,sample_1)

# get names of sample:
names = ro.r('names')
feature_name = names(sample_2)

# export as Rgrain:
grain_model = bnlearn.as_grain(alarm)
names = grain_model.names
grain_model1 = grain.compile_grain(grain_model, propagate = False, root =NULL, control = grain_model[7], details = 0)
output = grain.querygrain(grain_model1, nodes=feature_name, type="marginal")

fp = open('original_marginal_probability.txt', 'w')
fp.write(str(output))
fp.close()

# export as Rgrain:
grain_model = bnlearn.as_grain(model_1_fit)
names = grain_model.names
grain_model1 = grain.compile_grain(grain_model, propagate = False, root =NULL, control = grain_model[7], details = 0)
output = grain.querygrain(grain_model1, nodes=feature_name, type="marginal")

fp = open('sample_50k_marginal_probability.txt', 'w')
fp.write(str(output))
fp.close()


grain_model = bnlearn.as_grain(model_2_fit)
names = grain_model.names
grain_model1 = grain.compile_grain(grain_model, propagate = False, root =NULL, control = grain_model[7], details = 0)
output = grain.querygrain(grain_model1, nodes=feature_name, type="marginal")

fp = open('sample_500k_marginal_probability.txt', 'w')
fp.write(str(output))
fp.close()

# change topology (working)

data = bnlearn.rbn(alarm, n=5000)
o_model = bnlearn.drop_arc(o_model, "LVFAILURE", "LVEDVOLUME")
o_model = bnlearn.drop_arc(o_model, "LVEDVOLUME", "PCWP")
o_model = bnlearn.set_arc(o_model, "PCWP", "LVEDVOLUME")
o_model = bnlearn.drop_arc(o_model, "HYPOVOLEMIA", "LVEDVOLUME")

# replace parameters (not able to update individual probablities)
"""
c1 = ro.r('c(0.41138, 0.09938, 0.48924)')
c2 = ro.r('c("HIGH", "LOW", "NORMAL")')
list_r = ro.r('list')
matrix = ro.r('matrix')
table = ro.r('as.table')
cpt = matrix(c1,byrow = True, ncol = 3, dimnames = list_r(NULL, c2))
"""

model_fit = bnlearn.bn_fit(o_model, data)
#model_fit[4] = table(cpt)

grain_model = bnlearn.as_grain(model_fit)
names = grain_model.names
grain_model1 = grain.compile_grain(grain_model, propagate = False, root =NULL, control = grain_model[7], details = 0)
output = grain.querygrain(grain_model1, nodes=feature_name, type="marginal")

grdevices = rpackages.importr('grDevices')
pdf = ro.r('pdf')
pdf("modified_topology.pdf")
#actual plot
graph = bnlearn.graphviz_plot(o_model)
grdevices.dev_off()

fp = open('new_topo.txt', 'w')
fp.write(str(output))
fp.close()

#print(score)
print("fit model")