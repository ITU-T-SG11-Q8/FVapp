Configuration
  Environment variable
    - PYTHONUNBUFFERED=1;PYTHONPATH=%PYTHONPATH%\\\;\;[Working Directory]\\;[Working Directory]\fomm

Execute parameter
  Server 
    --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --is-server --listen-port [LISTEN_PORT] --keyframe-period 11000

  Client
    --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --server-ip [SERVER_IP] --server-port [SERVER_PORT] --keyframe-period 11000

