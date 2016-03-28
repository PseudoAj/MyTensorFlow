import matplotlib.pyplot as plt
a=[10,30,60,120]
b=[92.2,94.18,94.33,94.10]
plt.title('Accuracy vs Hidden Layers')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy of the Model')
plt.plot(a,b)
plt.savefig("acc.png")
plt.show()