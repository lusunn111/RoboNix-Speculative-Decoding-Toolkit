"""Run an original KERV script from the import-compatible vendor snapshot."""

from __future__ import annotations

import argparse

try:
    from speculative_decoding_service.service_bootstrap import run_vendor_script
except ModuleNotFoundError as exc:
    if exc.name != "speculative_decoding_service":
        raise
    # Support an independently extracted toolkit whose current directory is
    # speculative_decoding_service/ rather than its parent monorepo.
    from service_bootstrap import run_vendor_script


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Path relative to vendor/openvla")
    args, forwarded = parser.parse_known_args()
    import sys

    sys.argv = [args.script, *forwarded]
    run_vendor_script(args.script)


if __name__ == "__main__":
    main()
