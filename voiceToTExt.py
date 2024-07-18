import speech_recognition as sr

# Create a recognizer instance
recognizer = sr.Recognizer()

# Use the default microphone as the audio source
microphone = sr.Microphone()

print("Listening...")

# Adjust microphone sensitivity to ambient noise level
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

# Continuously listen for audio and convert it to text
while True:
    with microphone as source:
        try:
            audio_data = recognizer.listen(source, timeout=5)  # Adjust timeout as needed
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Error: Could not request results from Google Speech Recognition service; {0}".format(e))
