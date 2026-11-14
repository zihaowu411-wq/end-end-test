"""Use a separate transport node for blocking requests and no subscriptions."""

import sys

from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node


def serve():
    node = Node()
    for line in sys.stdin:
        service, request_type, payload = line.rstrip("\n").split(" ", 2)
        try:
            ok, message = node.request_raw(service, bytes.fromhex(payload), request_type,
                                           Boolean.DESCRIPTOR.full_name, 3000)
            print("OK" if ok and Boolean.FromString(message).data else "request timed out or rejected",
                  flush=True)
        except Exception as exc:
            print(str(exc).replace("\n", " "), flush=True)


if __name__ == "__main__":
    serve()
