from rapidfuzz import fuzz

def wake_up_detection_stub(ip:str) -> bool:
    ip = ip.lower()
    phrase = 'melody'
        
    partial_score = fuzz.partial_token_ratio(ip, phrase)
    if partial_score >= 85:
        print(f'Passed, Score: {partial_score}%, phrase: "{phrase}"')
        return True
    else:
        return False