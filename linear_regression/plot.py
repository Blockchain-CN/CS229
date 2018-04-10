import matplotlib.pyplot as plt

migtime = [16.6,16.5,16.9,17.1,17.2,18.1,18.2,18.4,19.4,19.3,20.4,20.3,22,21.7,22]
delay = [108,98,92,83,87,77,85,48,31,58,35,43,36,31,19]

plt.plot(migtime,"x-",label="migration time")
plt.plot(delay,"+-",label="request delay")

plt.show()