# Instruction PLD form:
## 1. Start the program:
  1). Click "Start" - "Anaconda Prompt"
  2). Copy the following code in the opened terminal:
     
     conda activate pld
     
     cd C:/Image/PLD_Workflow/
  
## 2. For parameter recording only: 

1). Start the form by copying the following code in the terminal:

      python pld_app_parameter.py
      
2). Fill in growth conditions and customized information

3). After finishing all recording and converting all videos, click "Save Parameters", and "Save to HDF5 and Upload". It may take a longer time to finish. Local HDF5 file (plume images) and JSON file (condition) will be saved locally and uploaded to the cloud.
     
     
## 3. For parameter and plume recording: 
  
1). Start the form by copying the following code in the opened terminal:

      python pld_app_plume.py
      
2). Fill in growth conditions and customized information

3). For every ablation cycle (different targets with pre-ablation and ablation):

  1>. Click the button "Move Videos to Pre-ablation Folder" or "Move Videos to Ablation Folder."

  2>. Use "HPV-X Viewer" software on the desktop to convert the raw file to readable images: 
"File" -> "Convert" -> Find the directory labelled start with your growth id -> Select all and click "CONVERT" 

  3>. Waiting time depends on how many videos are selected.

4). After finishing all recording and converting all videos, click "Save Parameters", and "Save to HDF5 and Upload". It may take a longer time to finish.


## 4. Addition instruction for camera position calibration:

1). Open Software "HPV-X" on desktop

2). Click the "Live" button and increase "EXPOSE" to 10,000,000ns to align the camera focus between the target and substrate holder.

3). Decrease the "EXPOSE" to 2,000,000ns and Click "REC" to start recording before ablation.
