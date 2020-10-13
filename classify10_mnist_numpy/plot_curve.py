import numpy as np


a = [i for i in range(10)]


# plot
import matplotlib.pyplot as plt
print(np.arange(1, len(a)+1))

# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# plt.plot(np.arange(1, len(train_batch_loss_list)+1), train_batch_loss_list)
# plt.xlabel('step')
# plt.ylabel('loss')
# plt.legend(['loss'])
# plt.subplot(122)
# plt.plot(np.arange(1, len(train_batch_acc_list)+1), train_batch_acc_list)
# plt.xlabel('step')
# plt.ylabel('accuracy')
# plt.legend(['accuracy'])
plt.savefig('train_test_batch_loss_acc.png')
# plt.show()

# train_loss_list = [0.2481435103262579, 0.12038503451325218, 0.10003320435681727, 0.08905439642865028, 0.08225027748247256, 0.07818226061337874, 0.07402232631754689, 0.07211302781825629, 0.07088991161426851, 0.06792148972527891]
# test_loss_list = [0.22569119116472358, 0.21431740568174537, 0.1308542627610702, 0.14264654327158918, 0.13055977254826429, 0.1375220185815396, 0.10885461573354828, 0.15326622025792966, 0.10939091855245359, 0.11130504089082414]
# train_acc_list = [0.9242, 0.9651666666666666, 0.97085, 0.9749, 0.9769333333333333, 0.9785166666666667, 0.9798, 0.9803833333333334, 0.98055, 0.9817333333333333]
# test_acc_list = [0.9248, 0.9304, 0.9571, 0.9547, 0.9579, 0.956, 0.9646, 0.9519, 0.9642, 0.9633]
#
# plt.figure(figsize=(12, 5))
# plt.subplot(121)
# plt.plot(np.arange(1, len(train_loss_list)+1), train_loss_list)
# plt.plot(np.arange(1, len(test_loss_list)+1), test_loss_list)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train_loss', 'test_loss'])
# plt.subplot(122)
# plt.plot(np.arange(1, len(train_acc_list)+1), train_acc_list)
# plt.plot(np.arange(1, len(test_acc_list)+1), test_acc_list)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['train_acc', 'test_acc'])
# # plt.show()
# plt.tight_layout()
# plt.savefig('train_test_loss_acc.png')
