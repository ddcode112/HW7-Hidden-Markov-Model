import matplotlib.pyplot as plt

x = [10, 100, 1000, 10000]
x_labels = [10, 100, 1000, 10000]
train = [-80.54472148949858, -74.9944493589361, -67.59317395746352, -60.6122844422585]
validation = [-87.97153090660584, -80.83229271827639, -70.45566558750309, -61.08562092334498]

plt.plot(x, train, label="Train Average Log-Likelihood")
plt.plot(x, validation, label="Validation Average Log-Likelihood")
plt.legend(loc='lower right')
plt.title('# Sequences used for training - Average Log-Likelihood')
plt.ylabel('Average Log-Likelihood')
plt.xlabel('# Sequences used for training')
plt.xticks(ticks=x, labels=x_labels, rotation=50)
plt.subplots_adjust(bottom=0.2)
plt.savefig('5_2.png')
