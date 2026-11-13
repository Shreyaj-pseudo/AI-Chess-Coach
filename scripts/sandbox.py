"""Start the local browser sandbox: python -m scripts.sandbox."""
import argparse

from sandbox_server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"Chess sandbox: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.sandbox.close()


if __name__ == "__main__":
    main()
