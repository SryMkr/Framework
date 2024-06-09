import numpy as np
import matplotlib.pyplot as plt

def forgetting_parameters(timing_steps, excel_start=0.9, excel_end=0.4, noise_start=0.1, noise_end=1, decay_rate=2):
    """create the weight pair of excellent memory and noise"""
    timing_points = np.linspace(0, 1, timing_steps)
    excel_list = (excel_start - excel_end) * np.exp(-decay_rate * timing_points) + excel_end
    excel_list = np.insert(excel_list, 0, 1)
    noise_list = (noise_end - noise_start) * (1 - np.exp(-decay_rate * timing_points)) + noise_start
    noise_list = np.insert(noise_list, 0, 0)
    return excel_list, noise_list


# Generate the data
timing_steps_num = 60
excel, noise = forgetting_parameters(timing_steps_num)

# Create a range for the x-axis
timing_points_list = np.arange(timing_steps_num + 1)  # +1 because of the np.insert adding one extra element

# Plot the data
plt.figure(figsize=(10, 6))

plt.plot(timing_points_list, excel, label='Excellent Memory', marker='o')
plt.plot(timing_points_list, noise, label='Noise', marker='x')

plt.title('Forgetting Parameters: Excellent Memory and Noise Over Time')
plt.xlabel('Timing Points')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()