
import numpy as np


my_info = list(input().split())
initial_v = float(my_info[0])
launch_angle = float(my_info[1])
init_x = float(my_info[2])
init_y = float(my_info[3])
time = float(my_info[4])
angle_radians = np.radians(launch_angle)
init_x_velocity = initial_v * np.cos(angle_radians)
init_y_velocity = initial_v * np.sin(angle_radians)
final_x = init_x + init_x_velocity * time
final_y = init_y +init_y_velocity * time + (-9.80665) * (time**2)/2
print(str(final_x)+ " " + str(final_y))
