import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 51))

# SimCLR losses
simclr_loss = [
    3.5346, 3.2420, 3.1575, 3.0917, 3.0604, 3.0357, 3.0163, 3.0004, 2.9832, 2.9726,
    2.9583, 2.9508, 2.9443, 2.9353, 2.9304, 2.9229, 2.9212, 2.9140, 2.9070, 2.9042,
    2.8995, 2.8958, 2.8937, 2.8887, 2.8855, 2.8801, 2.8770, 2.8745, 2.8693, 2.8662,
    2.8649, 2.8645, 2.8620, 2.8576, 2.8568, 2.8523, 2.8526, 2.8473, 2.8460, 2.8429,
    2.8401, 2.8372, 2.8368, 2.8346, 2.8334, 2.8293, 2.8322, 2.8298, 2.8283, 2.8249
]

# BYOL losses
byol_loss = [
    0.1144, 0.1666, 0.0981, 0.0691, 0.0616, 0.0548, 0.0500, 0.0470, 0.0426, 0.0453,
    0.0476, 0.0452, 0.0428, 0.0419, 0.0412, 0.0455, 0.0429, 0.0424, 0.0418, 0.0403,
    0.0400, 0.0419, 0.0417, 0.0439, 0.0405, 0.0411, 0.0404, 0.0315, 0.0356, 0.0385,
    0.0385, 0.0402, 0.0395, 0.0383, 0.0386, 0.0389, 0.0372, 0.0389, 0.0384, 0.0391,
    0.0385, 0.0406, 0.0397, 0.0408, 0.0409, 0.0386, 0.0374, 0.0367, 0.0348, 0.0346
]

# MoCo losses
moco_loss = [
    9.2069, 9.1904, 9.1808, 9.1699, 9.1648, 9.1639, 9.1659, 9.1681, 9.1661, 9.1636,
    9.1594, 9.1623, 9.1608, 9.1633, 9.1651, 9.1657, 9.1675, 9.1611, 9.1623, 9.1731,
    9.1641, 9.1664, 9.1765, 9.1725, 9.1747, 9.1771, 9.1707, 9.1692, 9.1681, 9.1689,
    9.1659, 9.1666, 9.1736, 9.1732, 9.1733, 9.1724, 9.1752, 9.1769, 9.1756, 9.1712,
    9.1731, 9.1716, 9.1730, 9.1713, 9.1665, 9.1688, 9.1765, 9.1648, 9.1666, 9.1672
]

# Plotting
plt.figure(figsize=(12*2, 6*2))
plt.plot(epochs, simclr_loss, label='SimCLR', marker='o', linestyle='-', markersize=10, linewidth=2)
plt.plot(epochs, moco_loss, label='MoCo', marker='x', linestyle='-', markersize=10, linewidth=2)
plt.plot(epochs, byol_loss, label='BYOL', marker='s', linestyle='-', markersize=10, linewidth=2)
plt.xlabel('Epoch', fontsize=40)
plt.ylabel('Training Loss', fontsize=40)
plt.title('Training Loss Comparison Over 50 Epochs', fontsize=40)
plt.legend(fontsize=40 , loc='best')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
