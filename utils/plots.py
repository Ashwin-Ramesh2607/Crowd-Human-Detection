import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
matplotlib.rcParams.update({
    'text.color' : 'black',
    'axes.labelcolor' : 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'})


class RealTimePlots():
    def __init__(self):
        self.people_counts = []
        self.violation_counts = []
        self.time_axis = []

    def add_data(self, people_count, violation_count):
        self.people_counts.append(people_count)
        self.violation_counts.append(violation_count)
        self.time_axis.append(int(len(self.time_axis) + 1))

    def retrieve_plot(self):
        fig = plt.figure(figsize=(6, 6), dpi=90)
        plt.plot(self.time_axis, self.people_counts, 'b', label='Person Count')
        plt.plot(self.time_axis, self.violation_counts, 'r', label='Social Distance Violations')

        plt.legend(loc='upper right')
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Person Count', fontsize=16)
        fig.tight_layout()
        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.shape
