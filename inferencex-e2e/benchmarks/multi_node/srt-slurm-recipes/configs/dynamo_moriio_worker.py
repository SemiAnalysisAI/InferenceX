import runpy
import uuid


def register_protocol():
    from dynamo.vllm.kv_connector_protocols import (
        KV_CONNECTOR_PROTOCOLS,
        KvConnectorProtocol,
    )

    class MoriioReadProtocol(KvConnectorProtocol):
        produces_kv_transfer_params = True

        def prefill_request_kv_transfer_params(self):
            self.transfer_id = "tx-" + uuid.uuid4().hex
            return {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "transfer_id": self.transfer_id,
                "remote_dp_size": 1,
                "remote_engine_id": None,
                "remote_block_ids": None,
            }

        def decode_request_kv_transfer_params(self, prefill_response):
            params = prefill_response.kv_transfer_params
            if (
                not isinstance(params, dict)
                or params.get("transfer_id") != self.transfer_id
            ):
                raise ValueError("MoRIIO prefill did not return the minted transfer ID")
            if not params.get("do_remote_prefill") or not params.get(
                "remote_engine_id"
            ):
                raise ValueError("MoRIIO prefill response is missing its READ handoff")
            for field in (
                "remote_block_ids",
                "remote_host",
                "remote_handshake_port",
                "remote_notify_port",
            ):
                if not params.get(field):
                    raise ValueError(f"MoRIIO handoff missing {field}")
            return params

    KV_CONNECTOR_PROTOCOLS["MoRIIOConnector"] = MoriioReadProtocol


if __name__ == "__main__":
    register_protocol()
    runpy.run_module("dynamo.vllm", run_name="__main__")
