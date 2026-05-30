import cv2
from faceMeshModule import FaceMeshDetector

# Initialize camera and detector
cap = cv2.VideoCapture(0)  # Use 1 if 0 doesn't work
detector = FaceMeshDetector()

# Landmark IDs for left and right eyes
leftEyeIDs = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
rightEyeIDs = [263, 362, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

# Variables
ratioListLeft = []
ratioListRight = []
blinkCount = 0
counterTime = 0
colorLeft = (255, 0, 255)
colorRight = (255, 0, 255)

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to grab frame.")
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # Draw eye landmarks
        for id in leftEyeIDs:
            cv2.circle(img, face[id], 3, colorLeft, cv2.FILLED)
        for id in rightEyeIDs:
            cv2.circle(img, face[id], 3, colorRight, cv2.FILLED)

        # Left eye distances
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        vertLeft, _ = detector.findDistance(leftUp, leftDown)
        horLeft, _ = detector.findDistance(leftLeft, leftRight)

        # Right eye distances
        rightUp = face[386]
        rightDown = face[374]
        rightLeft = face[263]
        rightRight = face[362]
        vertRight, _ = detector.findDistance(rightUp, rightDown)
        horRight, _ = detector.findDistance(rightLeft, rightRight)

        # Draw lines for visualization
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 2)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 2)
        cv2.line(img, rightUp, rightDown, (0, 200, 0), 2)
        cv2.line(img, rightLeft, rightRight, (0, 200, 0), 2)

        # Compute ratios
        ratioLeft = (vertLeft / horLeft) * 100
        ratioRight = (vertRight / horRight) * 100

        # Smooth using last 3 frames
        ratioListLeft.append(ratioLeft)
        ratioListRight.append(ratioRight)
        if len(ratioListLeft) > 3:
            ratioListLeft.pop(0)
            ratioListRight.pop(0)

        ratioAvgLeft = sum(ratioListLeft) / len(ratioListLeft)
        ratioAvgRight = sum(ratioListRight) / len(ratioListRight)

        # Detect blink if **both eyes** are closed
        ratioAvg = (ratioAvgLeft + ratioAvgRight) / 2
        if ratioAvg < 33 and counterTime == 0:
            blinkCount += 1
            counterTime = 1
            colorLeft = colorRight = (0, 255, 0)  # Change color on blink

        if counterTime != 0:
            counterTime += 1
            if counterTime > 10:
                counterTime = 0
                colorLeft = colorRight = (255, 0, 255)

        # Display blink count
        cv2.putText(img, f'Blinks: {blinkCount}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Show camera feed
    cv2.imshow('Blink Detection', img)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()