import numpy as np
import matplotlib.pyplot as plt

# Read data from file
data = np.loadtxt('run.log', delimiter=',')
x = data[:, 0]  # First column
y = data[:, 1]  # Second column

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-')
plt.xlabel('Problem Size')
plt.ylabel('Computation Time (seconds)')
plt.title('Problem Size vs Computation Time')
x_ticks = np.arange(0, max(x)+128, 128)  # from 0 to max value, step 128
plt.xticks(x_ticks, rotation=45)  # rotate labels 45 degrees if needed
plt.grid(True)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
plt.savefig("plots/ex4.png")