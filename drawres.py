import os
import numpy as np
import matplotlib.pyplot as plt 

def extract_data(keyname, root='./saved_models'):
    log_file = os.path.join(root, keyname, 'val_perf.log')
    alllines = open(log_file,'r').readlines()
    class_dict = {}
    for singleline in alllines:
        linelist = singleline.replace('\n','').split(' ')
        # 50 class_idx_3: precision 0.30581784264185996
        epoch = int(linelist[0])
        class_id = linelist[1].replace('idx_','')
        metric = linelist[2].replace('_','')
        value = float(linelist[3])
        tmpkey = class_id+metric
        if tmpkey not in class_dict.keys():
            class_dict[tmpkey]=[[],[]]
        if epoch<=1000:
            class_dict[tmpkey][0].append(epoch)
            class_dict[tmpkey][1].append(value)
    return class_dict

def draw(data, name, legend):
    plt.figure()
    for dd in data:
        plt.plot(dd[name][0],dd[name][1],'*-')
    plt.legend(legend)
    plt.title(name)
    plt.xlabel('Training Epochs')
    plt.ylabel('Test recall')
    plt.savefig('figs/'+name+'.jpg')
    plt.savefig('figs/'+name+'.pdf')
    plt.close()

baseline=extract_data('kp-1_vote-3')
# kp256_vote0=extract_data('kp256_vote0')
# kp64vote0=extract_data('kp64_vote0')
# kp128_vote0=extract_data('kp128_vote0')
legend = [ 'Baseline (1495-Points)', '256-Points', '128-Points', '64-Points']
legend = []
for keyname in baseline.keys():
    # draw([baseline, kp256_vote0, kp128_vote0, kp64vote0], keyname, legend)
    draw([baseline], keyname, legend)
