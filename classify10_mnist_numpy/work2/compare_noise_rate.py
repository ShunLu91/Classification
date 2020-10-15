import matplotlib.pyplot as plt
import numpy as np

baseline_test_loss = [0.22435842674788622, 0.15456214335051333, 0.14385024247444078, 0.12664716030625056, 0.1309124219434237, 0.11864119875217054, 0.12113898878606921, 0.12333103133573559, 0.12289466078892976, 0.12516878936645404]
baseline_test_acc = [0.9257, 0.9483, 0.9531, 0.9587, 0.9577, 0.9614, 0.9612, 0.9592, 0.9609, 0.9587]

nr0_test_loss = [0.36661956533523377, 0.2959634850999158, 0.20130359854033364, 0.1773002416767407, 0.18881830159219695, 0.1898175898738203, 0.15303503087572679, 0.20971052710923718, 0.1496435655579731, 0.1432566399506986]
nr0_test_acc = [0.9121, 0.9286, 0.9502, 0.9541, 0.952, 0.9532, 0.9618, 0.9456, 0.9639, 0.9652]

nr1_test_loss = [0.783160402630381, 0.6214340286719338, 0.4467823355867331, 0.35860595544540186, 0.3689964771593639, 0.30945764839107204, 0.331784183367545, 0.3379397464845157, 0.2988321835264189, 0.3237254744070718]
nr1_test_acc = [0.8643, 0.8933, 0.9217, 0.9386, 0.9381, 0.948, 0.9443, 0.947, 0.9528, 0.9488]

nr2_test_loss = [1.1145023386006008, 1.0070851674297443, 0.8135949481604131, 0.7579805167710288, 0.7121073528193987, 0.6691418349905967, 0.6740312266836399, 0.6615783913100881, 0.6877200345896403, 0.6200551068384708]
nr2_test_acc = [0.8074, 0.8445, 0.8787, 0.8911, 0.9017, 0.9103, 0.9117, 0.9134, 0.9112, 0.9239]

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(np.arange(1, len(baseline_test_loss)+1), baseline_test_loss)
plt.plot(np.arange(1, len(baseline_test_loss)+1), nr0_test_loss)
plt.plot(np.arange(1, len(baseline_test_loss)+1), nr1_test_acc)
plt.plot(np.arange(1, len(baseline_test_loss)+1), nr2_test_loss)

plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend(['Baseline', 'NR0_EXP', 'NR1_EXP', 'NR2_EXP'])

plt.subplot(122)
plt.plot(np.arange(1, len(baseline_test_acc)+1), baseline_test_acc)
plt.plot(np.arange(1, len(baseline_test_acc)+1), nr0_test_acc)
plt.plot(np.arange(1, len(baseline_test_acc)+1), nr1_test_acc)
plt.plot(np.arange(1, len(baseline_test_acc)+1), nr2_test_acc)

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend(['Baseline', 'NR0_EXP', 'NR1_EXP', 'NR2_EXP'])
plt.show()
