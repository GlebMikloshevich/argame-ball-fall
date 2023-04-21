import cv2
import numpy as np
from Vector import Vector
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist


class Ball:

    def __init__(self, position: Vector, radius = 10, mass = 1, bounceness = 0.975) -> None:
        self.position = position
        self.radius = radius
        self.mass = mass
        self.inverse_mass = 0 if self.mass == 0 else 1 / self.mass 
        self.velocity = Vector(0, 0)
        self.bounceness = bounceness

    def __collision_detection_line(self, sub_image):
        non_black = (sub_image != 0)
        if non_black is None:
            print("is None")
            return True, 0, 0
        x, y = np.where(non_black)
        return x.size > 0, x, y

    def __penetration_resolution_line(self, ):
        ...

    def __get_sub_image(self, image):
        rounded = self.position.round()
        sub_image = image[rounded[1] - self.radius:rounded[1] + self.radius+1, 
                        rounded[0] - self.radius:rounded[0] + self.radius+1].copy()
        mask = np.zeros(sub_image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (self.radius, self.radius), self.radius, 255, -1)
        sub_image = cv2.bitwise_and(sub_image, mask)
        return sub_image

    def check_line_collide(self, image):
        rounded = self.position.round()
        sub_image = self.__get_sub_image(image)
        collision, x, y = self.__collision_detection_line(sub_image)
        
        display_debug_image = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2BGR)
        if collision:
            coords = np.column_stack((x, y))
            dist = cdist([[self.radius, self.radius]], coords, metric='euclidean')
            dist_idx = np.argmin(dist, axis = 1)
            closest_point = Vector(
                self.position[0] - self.radius + coords[dist_idx[0]][0], 
                self.position[1] - self.radius + coords[dist_idx[0]][1])
            penetration_vect = self.position - closest_point
            penetration_vect = Vector(penetration_vect[1], penetration_vect[0])
            self.position += penetration_vect.normalize() * (self.radius - penetration_vect.length())
            normal = penetration_vect.normalize()
            sep_vel = self.velocity * normal
            self.velocity += normal * -(sep_vel + sep_vel * self.bounceness)

            #display_debug_image[coords[dist_idx[0]][0], coords[dist_idx[0]][1]] = (0,0,255)
            
            #sub_image = self.__get_sub_image(image)

        #cv2.imshow("Ball", cv2.resize(display_debug_image, (300, 300), interpolation = cv2.INTER_AREA))
        #cv2.imshow("new_ball", cv2.resize(sub_image, (300, 300), interpolation = cv2.INTER_AREA))

    def __collision_detection_wall(self, wall):
        return True if (wall.closest_point(self) - self.position).length() - wall.width / 2 <= self.radius else False

    def __penetration_resolution_wall(self, wall):
        penetration_vect = self.position - wall.closest_point(self)
        self.position += penetration_vect.normalize() * (self.radius - (penetration_vect.length() - wall.width / 2))

    def __collision_response_wall(self, wall): # TODO: add bouncness
        normal = (self.position - wall.closest_point(self)).normalize()
        sep_vel = self.velocity * normal
        self.velocity += normal * -(sep_vel + sep_vel * self.bounceness)

    def check_walls_collide(self, walls):
        for wall in walls:
            if not self.__collision_detection_wall(wall): continue
            self.__penetration_resolution_wall(wall)
            self.__collision_response_wall(wall)
    
    def __collision_detection_ball(self, ball):
        return True if self.radius + ball.radius >= (self.position - ball.position).length() else False

    def __penetration_resolution_ball(self, ball):
        distance = self.position - ball.position
        penetration_length = self.radius + ball.radius - distance.length()
        penetration_resolution = distance.normalize() * (penetration_length / (self.inverse_mass + ball.inverse_mass))
        self.position += penetration_resolution * self.inverse_mass
        ball.position += penetration_resolution * -ball.inverse_mass

    def __collision_response_ball(self, ball):
        normal = (self.position - ball.position).normalize()
        relative_velocity = self.velocity - ball.velocity
        sep_velocity = relative_velocity * normal

        vsep_diff = -sep_velocity * min(self.bounceness, ball.bounceness) - sep_velocity
        impulse = vsep_diff / (self.inverse_mass + ball.inverse_mass)
        impulse_vec = normal * impulse

        #sep_velocity_vector = normal * -sep_velocity * self.bounceness

        self.velocity += impulse_vec * self.inverse_mass
        ball.velocity += impulse_vec * -ball.inverse_mass

    def check_balls_collide(self, index, balls):
        for i in range(index+1, len(balls)):
            if not self.__collision_detection_ball(balls[i]): continue
            self.__penetration_resolution_ball(balls[i])
            self.__collision_response_ball(balls[i])
            
    def check_collide(self, index, width, height, image, balls, walls):
        self.check_walls_collide(walls)
        self.check_balls_collide(index, balls)
        # for ball in balls:
        #     rounded = ball.position.round()
        #     for i in range(rounded[0] - ball.radius,rounded[0] + ball.radius+1):
        #         for j in range(rounded[1] - ball.radius,rounded[1] + ball.radius+1):
        #             if ((ball.radius - i)**2 + (ball.radius - j)**2) <= ball.radius ** 2:
        #                 image[i][j] = 0
        #if index == 0:
        self.check_line_collide(image)
            
    def update(self, index, gravity, width, height, image, balls, walls, delta):
        self.add_force(Vector(0, gravity), delta)
        self.check_collide(index, width, height, image, balls, walls)
        self.position += self.velocity

    def add_force(self, force, delta):
        self.velocity += force * delta
        #print(self.velocity)

    def draw(self, image):
        cv2.circle(image, self.position.round().values, self.radius, 255, -1)
