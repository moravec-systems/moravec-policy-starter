"""WebSocket message types."""

from __future__ import annotations

from enum import Enum


class MessageType(str, Enum):
    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    PREPARE_CASE = "prepare_case"
    PREPARE_CASE_ACK = "prepare_case_ack"
    RESET = "reset"
    RESET_RESULT = "reset_result"
    CALL = "call"
    CALL_RESULT = "call_result"
    TRIAL_END = "trial_end"
    TRIAL_END_ACK = "trial_end_ack"
    CLOSE = "close"
    ERROR = "error"


REQUEST_RESPONSE_PAIRS: dict[MessageType, MessageType] = {
    MessageType.HELLO: MessageType.HELLO_ACK,
    MessageType.PREPARE_CASE: MessageType.PREPARE_CASE_ACK,
    MessageType.RESET: MessageType.RESET_RESULT,
    MessageType.CALL: MessageType.CALL_RESULT,
    MessageType.TRIAL_END: MessageType.TRIAL_END_ACK,
}
