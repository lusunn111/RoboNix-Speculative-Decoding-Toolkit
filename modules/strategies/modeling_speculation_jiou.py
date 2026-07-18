try:
    from speculative_decoding_service.service_bootstrap import activate_vendor
except ModuleNotFoundError as exc:
    if exc.name != "speculative_decoding_service":
        raise
    from service_bootstrap import activate_vendor

activate_vendor()
from prismatic.extern.hf.modeling_speculation_jiou import *  # noqa: F401,F403,E402
