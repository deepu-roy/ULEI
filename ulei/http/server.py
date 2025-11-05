"""
FastAPI HTTP server for online evaluation data ingestion.

Provides REST endpoints for submitting production traffic samples
for continuous shadow evaluation and monitoring.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from ulei.core.evaluator import Evaluator
from ulei.core.schemas import DatasetItem, EvaluationReport, EvaluationSuite
from ulei.http.models import (
    BatchIngestRequest,
    BatchIngestResponse,
    EvaluationEvent,
    EventStatus,
    HealthResponse,
)
from ulei.http.queue import EventQueue, QueuedEvent
from ulei.utils.config import load_config

logger = logging.getLogger(__name__)


class ULEIServer:
    """ULEI HTTP ingestion server for online shadow evaluation."""

    def __init__(
        self, config_path: Optional[str] = None, suite_config: Optional[EvaluationSuite] = None
    ):
        """
        Initialize ULEI server.

        Args:
            config_path: Path to server configuration file
            suite_config: Pre-loaded evaluation suite configuration
        """
        self.config = self._load_server_config(config_path) if config_path else {}
        self.suite_config = suite_config
        self.evaluator = Evaluator()
        self.event_queue = EventQueue()
        self.processing_task: Optional[asyncio.Task] = None
        self.app = self._create_app()

        # Event status tracking
        self.event_statuses: Dict[str, EventStatus] = {}
        self.completed_reports: Dict[str, EvaluationReport] = {}

    def _load_server_config(self, config_path: str) -> Dict[str, Any]:
        """Load server configuration from file."""
        try:
            return load_config(config_path)
        except Exception as e:
            logger.error(f"Failed to load server config: {e}")
            return {}

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="ULEI HTTP Ingestion API",
            description="HTTP endpoints for online evaluation data ingestion",
            version="1.0.0",
        )

        # Add exception handlers
        @app.exception_handler(ValidationError)
        async def validation_exception_handler(request, exc):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "validation_error",
                    "message": "Invalid request data",
                    "details": exc.errors(),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        @app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": "http_error",
                    "message": exc.detail,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        # Register routes
        app.add_api_route("/v1/health", self.health_check, methods=["GET"])
        app.add_api_route("/v1/eval/events", self.ingest_event, methods=["POST"])
        app.add_api_route("/v1/eval/events/batch", self.ingest_events_batch, methods=["POST"])
        app.add_api_route("/v1/eval/status/{event_id}", self.get_event_status, methods=["GET"])
        app.add_api_route("/v1/reports/{run_id}", self.get_report, methods=["GET"])

        return app

    async def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            queue_size=self.event_queue.size(),
            processing_active=self.processing_task is not None and not self.processing_task.done(),
        )

    async def ingest_event(
        self, event: EvaluationEvent, background_tasks: BackgroundTasks
    ) -> JSONResponse:
        """
        Ingest single evaluation event for processing.

        Args:
            event: Evaluation event data
            background_tasks: FastAPI background tasks

        Returns:
            JSON response with event acceptance confirmation
        """
        try:
            # Generate unique event ID if not provided
            if not event.event_id:
                event.event_id = f"evt_{uuid.uuid4().hex[:12]}"

            # Validate event data
            dataset_item = self._event_to_dataset_item(event)

            # Queue event for processing
            queued_event = QueuedEvent(
                event_id=event.event_id,
                dataset_item=dataset_item,
                suite_name=event.suite_name or "production_monitoring",
                queued_at=datetime.utcnow(),
                priority=event.metadata.get("priority", 1) if event.metadata else 1,
            )

            await self.event_queue.enqueue(queued_event)

            # Update status tracking
            self.event_statuses[event.event_id] = EventStatus(
                event_id=event.event_id,
                status="queued",
                queued_at=datetime.utcnow(),
                results_available=False,
            )

            # Start background processing if not already running
            if not self.processing_task or self.processing_task.done():
                background_tasks.add_task(self._start_background_processing)

            logger.info(f"Event {event.event_id} queued for evaluation")

            return JSONResponse(
                status_code=202,
                content={
                    "event_id": event.event_id,
                    "queued_at": datetime.utcnow().isoformat(),
                    "estimated_processing_time": "1-5 minutes",
                    "queue_position": self.event_queue.size(),
                },
            )

        except ValidationError as e:
            logger.error(f"Validation error for event: {e}")
            raise HTTPException(status_code=400, detail="Invalid event data")
        except Exception as e:
            logger.error(f"Failed to ingest event: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def ingest_events_batch(
        self, batch_request: BatchIngestRequest, background_tasks: BackgroundTasks
    ) -> BatchIngestResponse:
        """
        Ingest multiple evaluation events in batch.

        Args:
            batch_request: Batch ingestion request
            background_tasks: FastAPI background tasks

        Returns:
            Batch processing response with acceptance/rejection counts
        """
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        accepted_count = 0
        rejected_count = 0
        rejected_events = []

        for idx, event in enumerate(batch_request.events):
            try:
                # Process each event similar to single ingestion
                if not event.event_id:
                    event.event_id = f"evt_{uuid.uuid4().hex[:12]}"

                dataset_item = self._event_to_dataset_item(event)

                queued_event = QueuedEvent(
                    event_id=event.event_id,
                    dataset_item=dataset_item,
                    suite_name=event.suite_name or "production_monitoring",
                    queued_at=datetime.utcnow(),
                    priority=event.metadata.get("priority", 1) if event.metadata else 1,
                )

                await self.event_queue.enqueue(queued_event)

                self.event_statuses[event.event_id] = EventStatus(
                    event_id=event.event_id,
                    status="queued",
                    queued_at=datetime.utcnow(),
                    results_available=False,
                )

                accepted_count += 1

            except Exception as e:
                rejected_count += 1
                rejected_events.append({"index": idx, "reason": str(e)})
                logger.warning(f"Rejected event at index {idx}: {e}")

        # Start background processing
        if accepted_count > 0 and (not self.processing_task or self.processing_task.done()):
            background_tasks.add_task(self._start_background_processing)

        logger.info(f"Batch {batch_id}: {accepted_count} accepted, {rejected_count} rejected")

        return BatchIngestResponse(
            batch_id=batch_id,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            rejected_events=rejected_events,
        )

    async def get_event_status(self, event_id: str) -> EventStatus:
        """
        Get processing status of an evaluation event.

        Args:
            event_id: Event identifier

        Returns:
            Event status information
        """
        if event_id not in self.event_statuses:
            raise HTTPException(status_code=404, detail="Event not found")

        return self.event_statuses[event_id]

    async def get_report(self, run_id: str, format: str = "json") -> Any:
        """
        Retrieve evaluation report by run ID.

        Args:
            run_id: Run identifier
            format: Response format (json or html)

        Returns:
            Evaluation report in requested format
        """
        if run_id not in self.completed_reports:
            raise HTTPException(status_code=404, detail="Report not found")

        report = self.completed_reports[run_id]

        if format == "html":
            # Generate HTML report (simplified for now)
            html_content = f"""
            <html>
            <head><title>Evaluation Report {run_id}</title></head>
            <body>
                <h1>Evaluation Report</h1>
                <p><strong>Run ID:</strong> {report.run_id}</p>
                <p><strong>Suite:</strong> {report.suite_name}</p>
                <p><strong>Total Items:</strong> {len(report.results)}</p>
                <h2>Aggregates</h2>
                <ul>
                {chr(10).join(f"<li>{k}: {v:.3f}</li>" for k, v in report.aggregates.items())}
                </ul>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)

        return report

    def _event_to_dataset_item(self, event: EvaluationEvent) -> DatasetItem:
        """Convert evaluation event to dataset item."""
        return DatasetItem(
            id=event.event_id,
            input=event.input,
            output=event.output,
            reference=event.reference,
            context=event.input.get("context") if isinstance(event.input, dict) else None,
            metadata=event.metadata or {},
        )

    async def _start_background_processing(self):
        """Start background processing task."""
        if not self.processing_task or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_events())

    async def _process_events(self):
        """Background task to process queued events."""
        logger.info("Starting background event processing")

        while True:
            try:
                # Process events in batches
                batch = await self.event_queue.dequeue_batch(batch_size=10, timeout=30.0)

                if not batch:
                    # No events to process, sleep briefly
                    await asyncio.sleep(5)
                    continue

                logger.info(f"Processing batch of {len(batch)} events")

                # Group events by suite
                events_by_suite = {}
                for event in batch:
                    suite_name = event.suite_name
                    if suite_name not in events_by_suite:
                        events_by_suite[suite_name] = []
                    events_by_suite[suite_name].append(event)

                # Process each suite group
                for suite_name, suite_events in events_by_suite.items():
                    await self._evaluate_event_batch(suite_name, suite_events)

            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                await asyncio.sleep(10)  # Backoff on errors

    async def _evaluate_event_batch(self, suite_name: str, events: List[QueuedEvent]):
        """Evaluate a batch of events for a specific suite."""
        try:
            # Update event statuses to processing
            for event in events:
                if event.event_id in self.event_statuses:
                    self.event_statuses[event.event_id].status = "processing"
                    self.event_statuses[event.event_id].started_at = datetime.utcnow()

            # Load or use default suite configuration
            suite_config = self.suite_config
            if not suite_config:
                # Use default configuration for production monitoring
                from ulei.core.schemas import EvaluationSuite, MetricSpec

                suite_config = EvaluationSuite(
                    name=suite_name,
                    metrics=[MetricSpec(name="faithfulness"), MetricSpec(name="answer_relevancy")],
                    provider_priority=["ragas", "deepeval"],
                )

            # Convert events to dataset items
            dataset_items = [event.dataset_item for event in events]

            # Run evaluation
            report = await self.evaluator.run_evaluation(suite=suite_config, dataset=dataset_items)

            # Store completed report
            self.completed_reports[report.run_id] = report

            # Update event statuses to completed
            for event in events:
                if event.event_id in self.event_statuses:
                    self.event_statuses[event.event_id].status = "completed"
                    self.event_statuses[event.event_id].completed_at = datetime.utcnow()
                    self.event_statuses[event.event_id].results_available = True

            logger.info(f"Completed evaluation for {len(events)} events, run_id: {report.run_id}")

        except Exception as e:
            logger.error(f"Failed to evaluate event batch: {e}")

            # Update event statuses to failed
            for event in events:
                if event.event_id in self.event_statuses:
                    self.event_statuses[event.event_id].status = "failed"
                    self.event_statuses[event.event_id].completed_at = datetime.utcnow()
                    self.event_statuses[event.event_id].error_message = str(e)

    async def start(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """
        Start the HTTP server.

        Args:
            host: Host to bind to
            port: Port to bind to
            workers: Number of worker processes
        """
        config = uvicorn.Config(
            app=self.app, host=host, port=port, workers=workers, log_level="info"
        )

        server = uvicorn.Server(config)

        # Start background processing
        await self._start_background_processing()

        logger.info(f"Starting ULEI server on {host}:{port}")
        await server.serve()

    async def stop(self):
        """Stop the server and background tasks."""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        logger.info("ULEI server stopped")


def create_app(
    config_path: Optional[str] = None, suite_config: Optional[EvaluationSuite] = None
) -> FastAPI:
    """
    Factory function to create FastAPI app instance.

    Args:
        config_path: Path to server configuration
        suite_config: Pre-loaded evaluation suite

    Returns:
        Configured FastAPI application
    """
    server = ULEIServer(config_path, suite_config)
    return server.app
