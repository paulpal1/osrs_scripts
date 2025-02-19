import pygetwindow as gw
import cv2
import pyautogui
import numpy as np
import time
import random

# Image detection threshold
THRESHOLD = 0.9
RUNELITE_WINDOW = gw.getWindowsWithTitle('Runelite')[0]
R_LEFT, R_TOP, R_WIDTH, R_HEIGHT = RUNELITE_WINDOW.left, RUNELITE_WINDOW.top, RUNELITE_WINDOW.width, RUNELITE_WINDOW.height

# OVERLOADS
MAGE_OVERLOAD = cv2.imread('templates/mage_overload.PNG', cv2.IMREAD_GRAYSCALE)
OVERLOAD_EXPIRED = cv2.imread('templates/overload_expired.PNG', cv2.IMREAD_GRAYSCALE)
OVERLOADS = [cv2.imread('templates/overload_1.PNG', cv2.IMREAD_GRAYSCALE),
            cv2.imread('templates/overload_2.PNG', cv2.IMREAD_GRAYSCALE),
            cv2.imread('templates/overload_3.PNG', cv2.IMREAD_GRAYSCALE),
            cv2.imread('templates/overload_4.PNG', cv2.IMREAD_GRAYSCALE)]

# ABSORPTION POTIONS
ABS = [cv2.imread('templates/absorption_1.PNG', cv2.IMREAD_GRAYSCALE),
       cv2.imread('templates/absorption_2.PNG', cv2.IMREAD_GRAYSCALE),
       cv2.imread('templates/absorption_3.PNG', cv2.IMREAD_GRAYSCALE),
       cv2.imread('templates/absorption_4.PNG', cv2.IMREAD_GRAYSCALE)]

ROCKCAKE = cv2.imread('templates/rockcake.PNG', cv2.IMREAD_GRAYSCALE)

TARGET_HEALTH = cv2.imread('templates/target_health.PNG', cv2.IMREAD_GRAYSCALE)

PRAYER = cv2.imread('templates/prayer.PNG', cv2.IMREAD_GRAYSCALE)

EXIT = cv2.imread('templates/exit.PNG', cv2.IMREAD_GRAYSCALE)

# POWERUPS
ZAPPER = cv2.imread('templates/zapper.PNG', cv2.IMREAD_GRAYSCALE)
DAMAGE = cv2.imread('templates/damage.PNG', cv2.IMREAD_GRAYSCALE)
POWER = cv2.imread('templates/power.PNG', cv2.IMREAD_GRAYSCALE)
ULTIMATE = cv2.imread('templates/ultimate.PNG', cv2.IMREAD_GRAYSCALE)
POWERUPS = [ZAPPER, DAMAGE, POWER, ULTIMATE]

class Coord:
    def __init__(self, x, y, width, height, confidence):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

def getAbsorptionPotionPT(gameWindow):
    return getMostConfidentFromList(gameWindow, ABS)

def getOverloadPotionPT(gameWindow):
    return getMostConfidentFromList(gameWindow, OVERLOADS)

def getMostConfidentFromList(gameWindow, templates, threshold=THRESHOLD) -> Coord:
    """
    Returns the most confident match given a list of templates
    
    Args:
        gameWindow: The game window image
        templates: A list of templates to search for
        threshold: The confidence threshold to consider a match
    
    Returns:
        Coord: The most confident match
    """

    highestConfidence = 0.0
    bestCoord = None
    for template in templates:
        result = getMostConfidentMatch(gameWindow, template, threshold)
        if result is None:
            continue
        if result.confidence > highestConfidence:
            bestCoord = result
            highestConfidence = result.confidence

    if highestConfidence == 0.0:
        return None
    
    return bestCoord

def getMostConfidentMatch(gameWindow, template, threshold=THRESHOLD) -> Coord:
    """
    Returns the most confident match for a given template
    
    Args:
        gameWindow: The game window image
        template: The template to search for
        threshold: The confidence threshold to consider a match
    
    Returns:
        Coord: The most confident match
    """

    # Perform template matching
    result = cv2.matchTemplate(gameWindow, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    if not locations[0].size:
        return None
    
    # Get the best match (highest confidence score)
    best_match_index = np.argmax(result[locations])
    best_match_position = (locations[1][best_match_index], locations[0][best_match_index])  # (x, y)
    best_confidence = result[locations][best_match_index]

    # Calculate the center of the match
    template_height, template_width = template.shape
    center_x = best_match_position[0] + (template_width // 2)
    center_y = best_match_position[1] + (template_height // 2)

    return Coord(
        x=center_x, 
        y=center_y, 
        width=template_width,
        height=template_height,
        confidence=best_confidence)
    
def isOverloadExpired(gameWindow):
    mageOverload = getMostConfidentMatch(gameWindow, MAGE_OVERLOAD)
    if mageOverload is not None:
        return False
    
    return True

def atTargetHealth(gameWindow):
    return getMostConfidentMatch(gameWindow, TARGET_HEALTH) is not None

def atExit(gameWindow):
    return getMostConfidentMatch(gameWindow, EXIT) is not None

def getPrayerPT(gameWindow):
    return getMostConfidentMatch(gameWindow, PRAYER)

def powerUpAvailable(gameWindow):
    return getMostConfidentFromList(gameWindow, POWERUPS, 0.9) is not None

def getPowerUpPT(gameWindow):
    return getMostConfidentFromList(gameWindow, POWERUPS, 0.9)

def getRockCakePT(gameWindow):
    return getMostConfidentMatch(gameWindow, ROCKCAKE)

def randomSleep():
    min_duration = 0.1
    max_duration = 0.5
    random_duration = np.random.uniform(min_duration, max_duration)
    time.sleep(random_duration)

def move(coord: Coord):
    """
    Moves to a coordinate. Uses randomization to simulate human movement.
    
    Args:
        coord: The coordinate to move to
    """

    min_travel_duration = 0.1
    max_travel_duration = 1.0
    travel_random_duration = np.random.uniform(min_travel_duration, max_travel_duration)

    min_click_duration = 0.1
    max_click_duration = 0.5
    click_random_duration = np.random.uniform(min_click_duration, max_click_duration)

    center_x = R_LEFT + coord.x
    center_y = R_TOP + coord.y

    width = (coord.width * 0.8) / 2
    height = (coord.height * 0.8) / 2

    x_random = center_x + np.random.uniform(-width, width)
    y_random = center_y + np.random.uniform(-height, height)

    pyautogui.moveTo(x_random, y_random, duration=travel_random_duration)
    time.sleep(click_random_duration)

def moveAndClick(coord: Coord):
    """
    Moves to a coordinate and clicks. Uses randomization to simulate human movement.
    
    Args:
        coord: The coordinate to move to and click
    """

    move(coord)
    randomSleep()
    pyautogui.click()

def moveAndDoubleClick(coord: Coord):
    """
    Moves to a coordinate and double clicks. Uses randomization to simulate human movement.
    
    Args:
        coord: The coordinate to move to and double click
    """

    move(coord)
    randomSleep()
    pyautogui.doubleClick()

while True:
    screenshot = pyautogui.screenshot(region=(RUNELITE_WINDOW.left, RUNELITE_WINDOW.top, RUNELITE_WINDOW.width, RUNELITE_WINDOW.height))
    img = np.array(screenshot)
    gameWindow = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if atExit(gameWindow):
        print('Exiting')
        quit()

    if powerUpAvailable(gameWindow):
        print('Powerup available')
        powerUpPT = getPowerUpPT(gameWindow)
        moveAndClick(powerUpPT)    
        time.sleep(2)

    if isOverloadExpired(gameWindow):
        print('Applying overload')
        overloadPT = getOverloadPotionPT(gameWindow)
        moveAndClick(overloadPT)
        time.sleep(10)
        
        for i in range(6):
            time.sleep(1.0)
            print('Applying rockcake')
            rockCakePT = getRockCakePT(gameWindow)
            moveAndClick(rockCakePT)
        # Apply a few absorption potions
        for i in range(1):
            print('Applying absorption')
            time.sleep(1.0)
            absPT = getAbsorptionPotionPT(gameWindow)
            moveAndClick(absPT)
        
        continue

    time.sleep(1.0)

        
