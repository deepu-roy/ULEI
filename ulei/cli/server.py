"""
CLI command for starting the HTTP ingestion server.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn

from ulei.core.schemas import EvaluationSuite
from ulei.http.models import ServerConfig
from ulei.http.server import ULEIServer
from ulei.utils.config import load_config
from ulei.utils.logging import configure_logging as setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to server configuration file",
)
@click.option(
    "--suite-config",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    help="Path to evaluation suite configuration file",
)
@click.option("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
@click.option("--port", "-p", default=8000, type=int, help="Port to bind to (default: 8000)")
@click.option(
    "--workers", "-w", default=1, type=int, help="Number of worker processes (default: 1)"
)
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Set logging level",
)
@click.option("--access-log", is_flag=True, help="Enable HTTP access logging")
def server(
    config: Optional[Path],
    suite_config: Optional[Path],
    host: str,
    port: int,
    workers: int,
    reload: bool,
    verbose: bool,
    log_level: str,
    access_log: bool,
) -> None:
    """
    Start the ULEI HTTP ingestion server for online shadow evaluation.

    The server provides REST endpoints for submitting evaluation events
    and retrieving results. Events are processed asynchronously in batches
    with configurable evaluation suites.

    Examples:

    \b
    # Start server with default configuration
    ulei server

    \b
    # Start with custom configuration
    ulei server --config server_config.yaml --suite-config suite.yaml

    \b
    # Start on custom host/port with verbose logging
    ulei server --host 127.0.0.1 --port 9000 --verbose

    \b
    # Start in development mode with auto-reload
    ulei server --reload --verbose
    """
    # Setup logging
    setup_logging(verbose or log_level.upper() == "DEBUG")

    try:
        # Load server configuration
        server_config = None
        if config:
            config_data = load_config(str(config))
            server_config = ServerConfig(**config_data)
            click.echo(f"✅ Loaded server configuration from {config}")

        # Load evaluation suite configuration
        suite = None
        if suite_config:
            suite_data = load_config(str(suite_config))
            suite = EvaluationSuite(**suite_data)
            click.echo(f"✅ Loaded evaluation suite: {suite.name}")

        # Override config values with CLI arguments
        if server_config:
            if host != "0.0.0.0":
                server_config.host = host
            if port != 8000:
                server_config.port = port
            if workers != 1:
                server_config.workers = workers
        else:
            # Create default config with CLI values
            server_config = ServerConfig(
                host=host,
                port=port,
                workers=workers,
                queue_max_size=1000,
                batch_size=100,
                batch_timeout_seconds=30,
                rate_limit_events_per_minute=1000,
                reports_retention_days=30,
                event_status_retention_days=7,
            )

        # Create and configure server
        ulei_server = ULEIServer(config_path=str(config) if config else None, suite_config=suite)

        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=ulei_server.app,
            host=server_config.host,
            port=server_config.port,
            workers=server_config.workers if not reload else 1,  # Single worker for reload
            reload=reload,
            log_level=log_level.lower(),
            access_log=access_log,
        )

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum: int, frame: Optional[object]) -> None:
            click.echo(f"\n🛑 Received signal {signum}, shutting down...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Display startup information
        click.echo("\n🚀 Starting ULEI HTTP Server")
        click.echo(f"   Host: {server_config.host}")
        click.echo(f"   Port: {server_config.port}")
        click.echo(f"   Workers: {server_config.workers}")
        click.echo(f"   Log Level: {log_level}")
        if suite:
            click.echo(f"   Evaluation Suite: {suite.name}")
            click.echo(f"   Metrics: {[m.name for m in suite.metrics]}")
        click.echo(
            f"\n📍 Server will be available at: http://{server_config.host}:{server_config.port}"
        )
        click.echo(f"📊 Health check: http://{server_config.host}:{server_config.port}/v1/health")
        click.echo(f"📚 API docs: http://{server_config.host}:{server_config.port}/docs")
        click.echo("\n⏳ Starting server...")

        # Start the server
        server_instance = uvicorn.Server(uvicorn_config)

        if reload:
            click.echo("🔄 Development mode: Auto-reload enabled")

        # Run the server
        asyncio.run(server_instance.serve())

    except KeyboardInterrupt:
        click.echo("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise click.ClickException(f"Server startup failed: {e}")


@click.command()
@click.option("--host", default="localhost", help="Server host (default: localhost)")
@click.option("--port", "-p", default=8000, type=int, help="Server port (default: 8000)")
@click.option(
    "--timeout", default=5.0, type=float, help="Request timeout in seconds (default: 5.0)"
)
def health(host: str, port: int, timeout: float) -> None:
    """
    Check the health status of a running ULEI server.

    Examples:

    \b
    # Check local server
    ulei health

    \b
    # Check server on different host/port
    ulei health --host 192.168.1.100 --port 9000
    """
    try:
        import requests
    except ImportError:
        click.echo("❌ The 'requests' library is required for health checks.")
        click.echo("Install it with: uv add requests")
        raise click.ClickException("Missing dependency: requests")

    try:
        url = f"http://{host}:{port}/v1/health"
        click.echo(f"🔍 Checking server health at {url}")

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        health_data = response.json()

        click.echo(f"✅ Server is {health_data.get('status', 'unknown')}")
        click.echo(f"   Version: {health_data.get('version', 'unknown')}")
        click.echo(f"   Timestamp: {health_data.get('timestamp', 'unknown')}")

        if "queue_size" in health_data:
            click.echo(f"   Queue Size: {health_data['queue_size']}")

        if "processing_active" in health_data:
            processing_status = "active" if health_data["processing_active"] else "idle"
            click.echo(f"   Processing: {processing_status}")

    except requests.exceptions.ConnectionError:
        click.echo(f"❌ Could not connect to server at {host}:{port}")
        raise click.ClickException("Server not reachable")
    except requests.exceptions.Timeout:
        click.echo(f"⏰ Request timed out after {timeout} seconds")
        raise click.ClickException("Health check timed out")
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Health check failed: {e}")
        raise click.ClickException(f"Health check error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during health check: {e}")
        raise click.ClickException(f"Health check failed: {e}")


@click.command()
@click.argument("event_file", type=click.Path(exists=True, path_type=Path))
@click.option("--host", default="localhost", help="Server host (default: localhost)")
@click.option("--port", "-p", default=8000, type=int, help="Server port (default: 8000)")
@click.option(
    "--batch-size", default=10, type=int, help="Batch size for event submission (default: 10)"
)
@click.option(
    "--delay", default=0.0, type=float, help="Delay between batches in seconds (default: 0.0)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def submit_events(
    event_file: Path, host: str, port: int, batch_size: int, delay: float, verbose: bool
) -> None:
    """
    Submit evaluation events from a file to the ULEI server.

    EVENT_FILE should be a JSONL file with one evaluation event per line.

    Examples:

    \b
    # Submit events from file
    ulei submit-events events.jsonl

    \b
    # Submit in smaller batches with delay
    ulei submit-events events.jsonl --batch-size 5 --delay 1.0
    """
    import json
    import time

    import requests

    setup_logging(verbose)

    try:
        # Read events from file
        events = []
        with open(event_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    event_data = json.loads(line)
                    events.append(event_data)
                except json.JSONDecodeError as e:
                    click.echo(f"⚠️  Invalid JSON on line {line_num}: {e}")
                    continue

        if not events:
            raise click.ClickException("No valid events found in file")

        click.echo(f"📄 Loaded {len(events)} events from {event_file}")

        # Submit events in batches
        base_url = f"http://{host}:{port}"
        submitted = 0
        failed = 0

        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]

            try:
                if len(batch) == 1:
                    # Submit single event
                    response = requests.post(
                        f"{base_url}/v1/eval/events", json=batch[0], timeout=30
                    )
                else:
                    # Submit batch
                    response = requests.post(
                        f"{base_url}/v1/eval/events/batch", json={"events": batch}, timeout=30
                    )

                response.raise_for_status()

                if response.status_code == 202:
                    submitted += len(batch)
                    if verbose:
                        click.echo(f"✅ Batch {i // batch_size + 1}: {len(batch)} events submitted")
                else:
                    failed += len(batch)
                    click.echo(
                        f"❌ Batch {i // batch_size + 1}: Unexpected status {response.status_code}"
                    )

                # Add delay between batches if specified
                if delay > 0 and i + batch_size < len(events):
                    time.sleep(delay)

            except Exception as e:
                failed += len(batch)
                click.echo(f"❌ Batch {i // batch_size + 1}: {e}")

        # Summary
        click.echo("\n📊 Submission Summary:")
        click.echo(f"   Total Events: {len(events)}")
        click.echo(f"   Submitted: {submitted}")
        click.echo(f"   Failed: {failed}")

        if failed > 0:
            raise click.ClickException(f"Failed to submit {failed} events")

        click.echo("✅ All events submitted successfully!")

    except Exception as e:
        logger.error(f"Failed to submit events: {e}")
        if not isinstance(e, click.ClickException):
            raise click.ClickException(f"Event submission failed: {e}")
        raise
