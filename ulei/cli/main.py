"""
Main CLI entry point for ULEI.
"""

import click

from ulei.cli.compare import compare, trend
from ulei.cli.run import run
from ulei.cli.server import health, server, submit_events


@click.group()
@click.version_option(version="0.1.0", prog_name="ulei")
def main() -> None:
    """
    ULEI - Unified LLM Evaluation Interface

    A provider-agnostic evaluation framework for LLM and RAG systems.
    Switch between evaluation providers (Ragas, DeepEval, etc.) with simple configuration.
    """
    pass


# Register commands
main.add_command(run)
main.add_command(compare)
main.add_command(trend)
main.add_command(server)
main.add_command(health)
main.add_command(submit_events)


if __name__ == "__main__":
    main()
