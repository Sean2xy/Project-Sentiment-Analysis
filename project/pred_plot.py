import matplotlib.pyplot as plt
import torch
epoch = 10
metrics = torch.load("metrics.pt")
train_loss, test_loss, dev_acc, dev_f1, dev_recall, dev_pre = metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5],
iters = range(epoch)
plt.plot(iters, train_loss, 'g', label='train_loss')

plt.plot(iters, test_loss, 'g', label='dev_loss')

plt.plot(iters, dev_acc, 'k', label='dev_acc')

plt.plot(iters, dev_f1, 'k', label='dev_f1')

plt.plot(iters, dev_recall, 'k', label='dev_recall')

plt.plot(iters, dev_pre, 'k', label='dev_pre')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()
print('Finished Training')

