import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

plt.figure(figsize=(8, 6))
try:
    df = pd.read_csv("bug_to_request.csv")
except FileNotFoundError:
    print(f"Error: File not found. Please check the filename and try again.")
    exit()



request = list(range(0,20000) ) # Example X data
number_of_MuCoRest = (df['MuCoRest'].dropna().to_list()[:20000])
print(number_of_MuCoRest)
plt.plot(request, number_of_MuCoRest, label='MuCoRest')


try:
    df = pd.read_csv("ARAT-RL.csv")
except FileNotFoundError:
    print(f"Error: File not found. Please check the filename and try again.")
    exit()
number_of_ARAT_RL = (df['ARAT-RL'].dropna().to_list()[:20000])
plt.plot(request, number_of_ARAT_RL, label='ARAT-RL')






# Set the Y axis to increments of 10, 20, 30, etc.
plt.yticks(np.arange(0, 130, 10))

# Label the axes
plt.xlabel('Number of Requests')
plt.ylabel('Number of Bugs Found')

# Show the legend
plt.legend()

# Show the grid
plt.grid(True)

# Display the plot
plt.show()
