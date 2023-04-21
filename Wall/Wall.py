import cv2
from Vector import Vector

class Wall():

    def __init__(self, start: Vector, end: Vector, width):
        self.start = start
        self.end = end
        self.width = width

    def length(self):
        return (self.end - self.start).length()

    def normalized_direction(self):
        return (self.end - self.start).normalize()

    def closest_point(self, ball):
        ball_to_wall_start = self.start - ball.position
        norm_dir = self.normalized_direction()
        if norm_dir * ball_to_wall_start > 0: return self.start
        ball_to_wall_end = ball.position - self.end
        if norm_dir * ball_to_wall_end > 0: return self.end
        
        closestDist = norm_dir * ball_to_wall_start
        closestVect = norm_dir * closestDist
        return self.start - closestVect

    def draw(self, image):
        cv2.line(image, self.start.values, self.end.values, 255, self.width)