import rospy
from std_msgs.msg import Empty
from controller import Robot, GPS, Motor, Lidar, InertialUnit
import os
import numpy as np
import time
import csv

# Goal and start points definitions
goal_points = {
    "tb30": (-4.5, 4.5), "tb31": (-4.5, 4.5), "tb32": (-4.5, 4.5), "tb33": (-4.5, 4.5),
    "tb34": (-4.5, 4.5), "tb35": (-4.5, 4.5), "tb36": (-4.5, 4.5), "tb37": (-4.5, 4.5),
    "tb38": (-4.5, 4.5), "tb39": (-4.5, 4.5)
}

start_points = {
    "tb30": (3.5, -4.5), "tb31": (3.5, -4.5), "tb32": (3.5, -4.5), "tb33": (3.5, -4.5),
    "tb34": (3.5, -4.5), "tb35": (3.5, -4.5), "tb36": (3.5, -4.5), "tb37": (3.5, -4.5),
    "tb38": (3.5, -4.5), "tb39": (3.5, -4.5)
}

alpha = 5
beta = 6
epsilon = 1e-6
obstacle_threshold = 0.3
goal_threshold = 0.17
move_points = [(x * 0.5, y * 0.5) for x in range(-10, 11) for y in range(-10, 11)]
max_update = 100

pheromone_levels = {point: 10.0 for point in move_points}

traversal_data = []

start_time = None
goal_reached = False
returning_to_start = False
start_position = None
iteration_count = 0
max_iterations = 2

def get_goal_position(robot_name):
    return goal_points.get(robot_name, None)

def get_start_position(robot_name):
    return start_points.get(robot_name, None)

def callback_start_signal(msg):
    global start_signal_received, start_time, start_position
    start_signal_received = True
    start_time = time.time()
    start_position = get_start_position(robot_name)

def calculate_pheromone_level(point):
    return pheromone_levels[point]

def calculate_probabilities(attractions):
    total = sum(attraction for attraction, _ in attractions)
    if total == 0:
        return []
    return [(attraction / total, node) for attraction, node in attractions]

def select_node_based_on_probability(probabilities):
    total_prob = sum(prob for prob, _ in probabilities)
    if total_prob == 0:
        return None
    
    cumulative_probs = np.cumsum([prob / total_prob for prob, _ in probabilities])
    rand_value = np.random.rand()
    
    for i, cumulative_prob in enumerate(cumulative_probs):
        if rand_value < cumulative_prob:
            return probabilities[i][1]

def calculate_dynamic_q0(distance_to_goal, max_distance=10):
    return 1 - (distance_to_goal / max_distance)

def ant_colony_move(current_position, goal_position, move_points, alpha, beta, lidar_ranges):
    attractions = []
    distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(current_position))
    q_0 = calculate_dynamic_q0(distance_to_goal + epsilon)

    for point in move_points:
        if not is_obstacle_in_path(current_position, point, lidar_ranges):
            pheromone_level = calculate_pheromone_level(point)
            distance_to_point = np.linalg.norm(np.array(point) - np.array(current_position))
            distance_to_goal_point = np.linalg.norm(np.array(point) - np.array(goal_position))
            heuristic_value = 1 / (distance_to_goal_point + epsilon)
            attraction = (pheromone_level**alpha) * (heuristic_value**beta) * (1 / (distance_to_point + epsilon))
            attractions.append((attraction, point))
    
    probabilities = calculate_probabilities(attractions)
    if not probabilities:
        return None

    q = np.random.rand()
    if q < q_0 and probabilities:
        selected_node = max(probabilities, key=lambda x: x[0])[1]
    else:
        selected_node = select_node_based_on_probability(probabilities)
    return selected_node

def is_obstacle_in_path(current_position, point, lidar_ranges):
    direction = np.arctan2(point[1] - current_position[1], point[0] - current_position[0])
    for i, distance in enumerate(lidar_ranges):
        if distance < obstacle_threshold:
            lidar_angle = -np.pi + i * (2 * np.pi / len(lidar_ranges))
            if abs(lidar_angle - direction) < np.pi / 18:
                return True
    return False

def move_towards_node(current_position, next_node, speed, lidar_ranges):
    if next_node is None:
        return 0, 0

    desired_direction = np.arctan2(next_node[1] - current_position[1], next_node[0] - current_position[0])
    
    imu_values = imu.getRollPitchYaw()
    current_orientation = imu_values[2]

    direction_diff = desired_direction - current_orientation
    direction_diff = (direction_diff + np.pi) % (2 * np.pi) - np.pi
    left_speed = speed
    right_speed = speed
    for i, distance in enumerate(lidar_ranges):
        if distance < obstacle_threshold:
            lidar_angle = -np.pi + i * (2 * np.pi / len(lidar_ranges))
            if -np.pi / 4 < lidar_angle < np.pi / 4:
                if lidar_angle < 0:
                    left_speed = speed
                    right_speed = -speed
                else:
                    left_speed = -speed
                    right_speed = speed
                return left_speed, right_speed
    
    if abs(direction_diff) < 0.1:
        left_speed = speed
        right_speed = speed
    elif direction_diff > 0:
        left_speed = speed * (1 - abs(direction_diff) / np.pi)
        right_speed = speed
    else:
        left_speed = speed
        right_speed = speed * (1 - abs(direction_diff) / np.pi)
    
    return left_speed, right_speed

def update_pheromone_levels_around(current_position):
    radius = 0.5
    for point in move_points:
        if np.linalg.norm(np.array(move_point) - np.array(current_position)) <= radius:
            pheromone_levels[point] += 1
            if pheromone_levels[point] > max_update:
                pheromone_levels[point] = max_update
            traversal_data.append((point, pheromone_levels[point]))

robot = Robot()
timestep = int(robot.getBasicTimeStep())
gps = robot.getDevice('gps')
gps.enable(timestep)
imu = robot.getDevice('imu')
imu.enable(timestep)
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)
lidar.enablePointCloud()
right_motor = robot.getDevice('right wheel motor')
left_motor = robot.getDevice('left wheel motor')
right_motor.setPosition(float('inf'))
left_motor.setPosition(float('inf'))
right_motor.setVelocity(0)
left_motor.setVelocity(0)

robot_name = os.getenv('WEBOTS_CONTROLLER_URL').split('/')[-1]

rospy.init_node(f'{robot_name}_controller')
rospy.Subscriber(f'/{robot_name}/start_signal', Empty, callback_start_signal)

start_signal_received = False

while not rospy.is_shutdown() and iteration_count < max_iterations:
    if robot.step(timestep) == -1:
        break

    if not start_signal_received:
        continue

    gps_value = gps.getValues()
    current_position = (gps_value[0], gps_value[1])
    goal_position = get_goal_position(robot_name)
    lidar_ranges = lidar.getRangeImage()

    if goal_reached:
        if returning_to_start:
            distance_to_start = np.linalg.norm(np.array(current_position) - np.array(start_position))
            if distance_to_start < goal_threshold:
                right_motor.setVelocity(0)
                left_motor.setVelocity(0)
                elapsed_time = time.time() - start_time
                print(f"Robot {robot_name} has returned to the start position. Time taken: {elapsed_time:.2f} seconds")
                iteration_count += 1
                returning_to_start = False
                goal_reached = False
                start_signal_received = True
                start_time = time.time()
                with open(f'{robot_name}_traversal_data.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Iteration', iteration_count])
                    csvwriter.writerow(['Node', 'Pheromone Level'])
                    for data in traversal_data:
                        csvwriter.writerow(data)
            else:
                next_node = ant_colony_move(current_position, start_position, move_points, alpha, beta, lidar_ranges)
                if next_node and np.linalg.norm(np.array(next_node) - np.array(current_position)) < 0.2:
                    update_pheromone_levels_around(current_position)
                left_speed, right_speed = move_towards_node(current_position, next_node, 6.28, lidar_ranges)
                right_motor.setVelocity(right_speed)
                left_motor.setVelocity(left_speed)
        else:
            returning_to_start = True
    else:
        if goal_position:
            distance_to_goal = np.linalg.norm(np.array(current_position) - np.array(goal_position))
            if distance_to_goal < goal_threshold:
                right_motor.setVelocity(0)
                left_motor.setVelocity(0)
                elapsed_time = time.time() - start_time
                print(f"Robot {robot_name} has reached the goal position. Time taken: {elapsed_time:.2f} seconds")
                goal_reached = True
            else:
                next_node = ant_colony_move(current_position, goal_position, move_points, alpha, beta, lidar_ranges)
                if next_node and np.linalg.norm(np.array(next_node) - np.array(current_position)) < 0.2:
                    update_pheromone_levels_around(current_position)
                left_speed, right_speed = move_towards_node(current_position, next_node, 6.28, lidar_ranges)
                right_motor.setVelocity(right_speed)
                left_motor.setVelocity(left_speed)

    if iteration_count >= max_iterations:
        # Log the traversal data and pheromone levels to a CSV file
        with open(f'{robot_name}_final_traversal_data.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Node', 'Pheromone Level'])
            for node, pheromone_level in pheromone_levels.items():
                csvwriter.writerow([node, pheromone_level])
        break
