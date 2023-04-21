import time

import numpy as np

from Ball import Ball
from Wall import Wall
from Vector import Vector


class Game():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.dt = 0
        self.gravity = 2
        self.balls = [Ball(Vector(i, 200), 10, 1) for i in range(50, 450, 30)]#[Ball(Vector(100, 200), 10)]#
        self.walls = [
        Wall(Vector(250, 200), Vector(400, 150), 10),
        Wall(Vector(100, 400), Vector(60, 100), 10)]
        self.create_edge_walls(width, height)

    def create_edge_walls(self, width, height):
        self.walls.extend([
            Wall(Vector(-5, -5), Vector(width + 5, -5), 10),
            Wall(Vector(width + 5, -5), Vector(width + 5, height + 5), 10),
            Wall(Vector(width + 5, height + 5), Vector(-5, height + 5), 10),
            Wall(Vector(-5, height + 5), Vector(-5, -5), 10)
        ])
        
    def update(self, binarize_image):
        """Calculate 1 frame"""
        start = time.perf_counter()

        for index in range(len(self.balls)):
            self.balls[index].update(index, self.gravity, self.width, self.height, binarize_image, self.balls, self.walls, self.dt)

        binarize_image *= 0

        for index in range(len(self.balls)):   
            self.balls[index].draw(binarize_image)

        for wall in self.walls:
            wall.draw(binarize_image)

        self.dt = (time.perf_counter() - start)*10

        return binarize_image

    def get_drawn_objects(self):
        empty_img = np.zeros((self.height,self.width,1), np.uint8)

        for ball in self.balls:
            ball.draw(empty_img)

        for wall in self.walls:
            wall.draw(empty_img)

        return empty_img
