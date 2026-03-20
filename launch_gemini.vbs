Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\ofekb\Documents\Personal Projects\gemini-search"
WshShell.Run "python app.py", 0, False
WScript.Sleep 2500
WshShell.Run "chrome.exe --app=http://localhost:5000", 1, False
