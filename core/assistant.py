
import datetime
import subprocess


def run_command(command: str) -> None:
    command = command.lower()

    if "open browser" in command:
        subprocess.Popen(["xdg-open", "https://www.google.com"])
        print("Opened Browser")
    elif "open code" in command:
        subprocess.Popen(["code"])  # VS Code
        print("Vs code is now open")
    elif "show time" in command:
        print(datetime.now())
    elif "open spotify" in command:
        subprocess.Popen(["spotify"])
        print("Spotify is now opened")
    elif "play music" in command:
        subprocess.run(["playerctl", "-p", "spotify", "play"], check=True)
        print("Playing music")
    else:
        print('Command not found')