import matplotlib.pyplot as plt
import os
import datetime


class Visualization:
    def __init__(self, path="plots", dpi=300):
        pass

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session
        and save the relative data to a text file and image.
        """
        min_val = min(data)
        max_val = max(data)

        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(filename)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        plt.grid(True)

        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save_path =  f"plots/{filename}_{timestamp}.jpg"
        # plt.savefig(save_path)

        plt.show()  # Show the plot
        plt.close()  # Close the figure after displaying

        # print(f"Plot saved at: {save_path}")
        print(f"Minimum value: {min_val}")
        print(f"Maximum value: {max_val}")
