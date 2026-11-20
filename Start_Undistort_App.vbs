Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = repoDir

pythonExe = fso.BuildPath(repoDir, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonExe) Then
    pythonExe = "pythonw.exe"
End If

args = "-m undistort_app"
For Each arg In WScript.Arguments
    args = args & " """ & Replace(arg, """", """""") & """"
Next

shell.Run """" & pythonExe & """ " & args, 0, False
