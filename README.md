Progress
- Forked from https://github.com/alievk/avatarify-python
- Implementation of communication module between server/client (completed, but planned to use peerApi module later)
- Apply encoding/decoding module using avatarify (Server and client must be run on different PCs, and cuda environment must be set for each)
- Transmit and receive key frames and avatarify feature points and display them on the screen
- Completion of screen composition using pyqt5
- Application of peerApi Prototype module completed (Implementation completed to apply the previously received content as a parameter classification)
- Screen output part after video encoding/decoding process is in progress
- telecommuication audio module processing 

Future progress
- Audio module processing (synchronization with video, etc. will be processed later)
- Payload module processing of transmitted and received packets (this part will be completed by the end of the year)
- Application of SNNM mode (Details of application period and contents will be confirmed after additional review)
- Applied when peerApi SDK module is completed (scheduled to be applied by the end of 2023)

Build Configuration
  * Environment variable
    - PYTHONUNBUFFERED=1;PYTHONPATH=%PYTHONPATH%\\\;\;[Working Directory]\\;[Working Directory]\fomm;[Working Directory]\SPIGA
  * Due to github capacity issues, the "vox-adv-cpk.pth.tar" file must be downloaded separately.

-- Goorooroomee App Test Procedure ---
Installation
1) install conda module
    - conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
2) install pip module
    - pip install tensorflow
    - pip install scikit-learn
    - pip install sort-tracker-py
    - pip install retinaface-py
    - pip install face-alignment
    - pip install pyqt5
    - pip install pyqt5-tools
    - pip install pygame
    - pip install pyaudio
    - pip install hp2p-api

1. Run FVApp as a channel creator
    parameter
	python main.py

	1) Create Channel
	2) Channel Join

2) Run FVApp as a channel participant
   parameter
	python main.py

	1) Channel Join

3) keyframe send (The default keyframe period is 2000ms, and you can set the duration through the --keyframe-period (ms) option.)
4) After receiving the keyframe and receiving the feature points, the video will be displayed