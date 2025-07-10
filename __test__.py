from rapidfuzz import fuzz

score = None
ip = None
wake_up_phrases = ['melody', 'Hi melody', 'yo melody']

while True:
    ip:str = input('Enter phrase: ')
    for phrase in wake_up_phrases:
        score = fuzz.ratio(ip, phrase)
        if score >= 80:
            print(f'Passed, Score: {score}%, phrase: "{phrase}"')