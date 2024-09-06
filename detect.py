import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model/emotion_detector_model.h5')
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_accuracy}')
# Define sign labels
sign_labels = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine'
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    reshaped = resized.reshape(1, 64, 64, 1) / 255.0

    # Make prediction
    prediction = model.predict(reshaped)
    class_idx = np.argmax(prediction)
    class_label = sign_labels.get(class_idx, 'Unknown')

    # Display the resulting frame
    cv2.putText(frame, class_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




