class Cow():
    def __init__(self, name):
        self.name = name
        self.age = 0
    def speak(self):
        print('Moo!')
    def getAge(self):
        print(self.age)
        return self.age
    def setAge(self, age):
        self.age = age

    def setAge2(self):
        change(self.age)


carla = Cow('Carla')
carla.speak()
i = carla.getAge()
new = 5
def change(i):
    i = new

carla.getAge()