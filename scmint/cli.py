import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(prog="scsaga")
    parser.add_argument("yaml_file", help="YAML input file with configuration")
    args = parser.parse_args()

    with open(args.yaml_file) as f:
        raw_cfg = yaml.safe_load(f)

    from scmint.scsaga import main as run
    run(raw_cfg)


if __name__ == "__main__":
    main()