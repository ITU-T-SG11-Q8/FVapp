"""
참조 문서 : https://docs.google.com/document/d/1Yz60FiMxJi8upZrMvffDTRNW4mUrCpUf

python3.8

@version 0.1
@author jaylee@jayb.kr
"""

from typing import Callable
from .classes import *
from .api_func import __CheckRequest, __Creation, __Query, __Modification, __Join, __SearchPeer, __SendData, __Leave, __Removal, __SetNotificatonListener

__all__ = [
    "Creation",
    "Query",
    "Modification",
    "Join",
    "SearchPeer",
    "SendData",
    "Leave",
    "Removal",
    "SetNotificatonListener",
]

def Creation(req: CreationRequest) -> CreationResponse:
    """
    신규 서비스 세션 생성
    """
    return __CheckRequest(req, __Creation)

def Query(overlayId: str = None, title: str = None, description: str = None) -> QueryResponse:
    """
    서비스 세션 목록 조회
    """
    return __Query(overlayId, title, description)

def Modification(req: ModificationRequest) -> Response:
    """
    서비스 세션 변경
    """
    return __CheckRequest(req, __Modification)

def Join(req: JoinRequest) -> JoinResponse:
    """
    서비스 세션 참가
    """
    return __CheckRequest(req, __Join)

def SearchPeer(req: SearchPeerRequest) -> SearchPeerResponse:
    """
    서비스 세션에 참가한 Peer 목록 조회
    """
    return __CheckRequest(req, __SearchPeer)

def SendData(req: SendDataRequest) -> Response:
    """
    Data broadcast
    """
    return __CheckRequest(req, __SendData)

def Leave(req: SendDataRequest) -> Response:
    """
    Data broadcast
    """
    return __CheckRequest(req, __Leave)

def Removal(req: RemovalRequest) -> Response:
    """
    서비스 세션 삭제
    """
    return __CheckRequest(req, __Removal)

def SetNotificatonListener(overlayId: str, peerId: str, func: Callable[[Notification], None]) -> bool:
    """
    세션 변경 내용 수신 설정
    """
    return __SetNotificatonListener(overlayId, peerId, func)