from anesthetic import NestedSamples
nested_root = '/home/isidro/Documents/gitHub/misRepos/nestedSampling/outputs/test'
nested = NestedSamples(root=nested_root)

ns_output = nested.ns_output()

nested.gui()

print("ns_output[:6]:\n{}\n".format(ns_output[:6]))

for x in ns_output:
    print('%10s = %9.2f +/- %4.2f' % (x, ns_output[x].mean(), ns_output[x].std()))
