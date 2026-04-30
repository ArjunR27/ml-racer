import math
class Car:
    """
    Represents the car object

    Attributes:
        x (float): The x position of the car
        y (float): The y position of the car
        direction (float): The direction the car is facing (-360 <= 0 <= 360 degrees)
        speed (float): The speed of the car ()
        tire_health: The health of the tire (0 - 1.0)
        alive: Has the car crashed or not
        dt: time step (fps)
    """
    ACCELERATION = 200 # pixels per second^2
    DRAG = 0.92 # air resistance modeling for how much speed you lose (8% speed loss every time step you let go of the throttle)
    
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = 0.0
        self.tire_health = 1.0
        self.alive = True
    
    def move(self, dt):
        angle = math.radians(self.direction)
        vx = math.cos(angle) * self.speed
        vy = -math.sin(angle) * self.speed
        self.x += (vx * dt)
        self.y += (vy * dt)
    
    def apply_throttle(self, throttle, dt):
        # speed = speed + (acceleration * time)
        self.speed += (throttle * self.ACCELERATION * dt)
        self.speed *= (self.DRAG ** dt)
        print(self.speed)

# example of the car moving from left to right across 30fps given that the car speed is 100 pixels/units per second --> 30 frames per second --> 30 frames to go 100 pixels
car = Car(x=0, y=0, direction=0)
car.speed = 100.0

for i in range(30):
    car.apply_throttle(0.0, dt=1/30)
    car.move(dt=1/30)
    print(f"step {i+1}: x={car.x:.2f}, y={car.y:.2f}")