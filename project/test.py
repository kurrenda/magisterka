
from ohbot import ohbot


try:
    ohbot.move(0,1,2)
    ohbot.wait(0.5)
    ohbot.move(0,2,2)
    ohbot.move(0,3,2)
except Exception as e:
    print(e)
# ohbot.wait(2)
# ohbot.move(1,4,3)