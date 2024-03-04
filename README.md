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

Execute parameter
  * Server  
    --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --is-server --listen-port [LISTEN_PORT] --keyframe-period 11000
    
  * Client  
    --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --server-ip [SERVER_IP] --server-port [SERVER_PORT] --keyframe-period 11000

* Due to github capacity issues, the "vox-adv-cpk.pth.tar" file must be downloaded separately.

-- Goorooroomee App Test Procedure ---
Precondition
1) torch.cuda.is_available() is True
2) Activate cuda on both Server/Client
3) Server and client must be composed of separate computers connected by a network.

1. Run GooroomeeApp as a server
    parameter
	--config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --is-server --listen-port [LISTEN_PORT] --keyframe-period 11000
	1) Create Channel
	2) Channel Join

2) Run GooroomeeApp as a client
   parameter
	--config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --server-ip [SERVER_IP] --server-port [SERVER_PORT] --keyframe-period 11000
	1) Channel Create (In future actual use, channel create will be used after executing once - JayB api applied)
	2) Channel Join

3) keyframe send (period conf is --keyframe-period 11000 (ms)
4) After receiving the keyframe and receiving the feature points, the video is displayed