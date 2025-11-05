#!/usr/bin/env python3
"""
HTTP Server API Testing Script

Tests all endpoints documented in docs/HTTP_API.md
"""

import asyncio
import json
import time
from typing import Any, Dict, List

import httpx


class ServerAPITester:
    """Test suite for ULEI HTTP API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results: List[Dict[str, Any]] = []

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if details:
            print(f"  {details}")
        self.test_results.append({"name": name, "passed": passed, "details": details})

    async def test_health_check(self):
        """Test GET /v1/health endpoint."""
        print("\n🔍 Testing Health Check Endpoint")
        print("=" * 60)

        try:
            response = await self.client.get(f"{self.base_url}/v1/health")

            self.log_test(
                "Health check returns 200",
                response.status_code == 200,
                f"Status: {response.status_code}",
            )

            data = response.json()
            required_fields = ["status", "version", "timestamp", "queue_size", "processing_active"]

            for field in required_fields:
                self.log_test(
                    f"Health response contains '{field}'",
                    field in data,
                    f"Value: {data.get(field)}",
                )

            self.log_test(
                "Health status is 'healthy'",
                data.get("status") == "healthy",
                f"Status: {data.get('status')}",
            )

            print(f"\nFull Response:\n{json.dumps(data, indent=2)}")

        except Exception as e:
            self.log_test("Health check endpoint", False, f"Error: {e}")

    async def test_single_event_ingestion(self):
        """Test POST /v1/eval/events endpoint."""
        print("\n🔍 Testing Single Event Ingestion")
        print("=" * 60)

        event_data = {
            "event_id": "test_evt_001",
            "suite_name": "server_shadow_evaluation",
            "input": {
                "query": "What is the capital of France?",
                "context": [{"text": "Paris is the capital of France.", "source_id": "doc_123"}],
            },
            "output": {"answer": "Paris is the capital of France.", "citations": ["doc_123"]},
            "reference": {"answer": "Paris"},
            "metadata": {"user_id": "test_user", "session_id": "test_session_001"},
        }

        try:
            response = await self.client.post(f"{self.base_url}/v1/eval/events", json=event_data)

            self.log_test(
                "Single event returns 202 Accepted",
                response.status_code == 202,
                f"Status: {response.status_code}",
            )

            data = response.json()
            required_fields = ["event_id", "queued_at", "estimated_processing_time"]

            for field in required_fields:
                self.log_test(
                    f"Response contains '{field}'", field in data, f"Value: {data.get(field)}"
                )

            self.log_test(
                "Event ID matches request",
                data.get("event_id") == "test_evt_001",
                f"Returned: {data.get('event_id')}",
            )

            print(f"\nFull Response:\n{json.dumps(data, indent=2)}")
            return data.get("event_id")

        except Exception as e:
            self.log_test("Single event ingestion", False, f"Error: {e}")
            return None

    async def test_batch_ingestion(self):
        """Test POST /v1/eval/events/batch endpoint."""
        print("\n🔍 Testing Batch Event Ingestion")
        print("=" * 60)

        batch_data = {
            "events": [
                {
                    "input": {"query": "What is ML?"},
                    "output": {"answer": "Machine Learning is a subset of AI..."},
                },
                {
                    "input": {"query": "What is AI?"},
                    "output": {"answer": "Artificial Intelligence is..."},
                },
                {
                    "input": {"query": "What is deep learning?"},
                    "output": {"answer": "Deep learning uses neural networks..."},
                },
            ]
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/v1/eval/events/batch", json=batch_data
            )

            self.log_test(
                "Batch ingestion returns 202",
                response.status_code == 202,
                f"Status: {response.status_code}",
            )

            data = response.json()
            required_fields = ["batch_id", "accepted_count", "rejected_count"]

            for field in required_fields:
                self.log_test(
                    f"Response contains '{field}'", field in data, f"Value: {data.get(field)}"
                )

            self.log_test(
                "All events accepted",
                data.get("accepted_count") == 3,
                f"Accepted: {data.get('accepted_count')}, Rejected: {data.get('rejected_count')}",
            )

            print(f"\nFull Response:\n{json.dumps(data, indent=2)}")

        except Exception as e:
            self.log_test("Batch ingestion", False, f"Error: {e}")

    async def test_event_status(self, event_id: str):
        """Test GET /v1/eval/status/{event_id} endpoint."""
        print("\n🔍 Testing Event Status Tracking")
        print("=" * 60)

        if not event_id:
            self.log_test("Event status check", False, "No event ID available")
            return

        try:
            response = await self.client.get(f"{self.base_url}/v1/eval/status/{event_id}")

            self.log_test(
                "Status check returns 200",
                response.status_code == 200,
                f"Status: {response.status_code}",
            )

            data = response.json()
            required_fields = ["event_id", "status", "queued_at", "results_available"]

            for field in required_fields:
                self.log_test(
                    f"Status response contains '{field}'",
                    field in data,
                    f"Value: {data.get(field)}",
                )

            valid_statuses = ["queued", "processing", "completed", "failed"]
            self.log_test(
                "Status is valid",
                data.get("status") in valid_statuses,
                f"Status: {data.get('status')}",
            )

            print(f"\nFull Response:\n{json.dumps(data, indent=2)}")

        except Exception as e:
            self.log_test("Event status check", False, f"Error: {e}")

    async def test_invalid_requests(self):
        """Test error handling with invalid requests."""
        print("\n🔍 Testing Error Handling")
        print("=" * 60)

        # Test missing required field
        try:
            invalid_event = {
                "input": {"query": "Test query"}
                # Missing 'output' field
            }

            response = await self.client.post(f"{self.base_url}/v1/eval/events", json=invalid_event)

            self.log_test(
                "Invalid event returns 400",
                response.status_code == 400,
                f"Status: {response.status_code}",
            )

            data = response.json()
            self.log_test(
                "Error response contains 'error' field",
                "error" in data,
                f"Error type: {data.get('error')}",
            )

            print(f"\nError Response:\n{json.dumps(data, indent=2)}")

        except Exception as e:
            self.log_test("Invalid request handling", False, f"Error: {e}")

        # Test non-existent event status
        try:
            response = await self.client.get(
                f"{self.base_url}/v1/eval/status/nonexistent_event_123"
            )

            self.log_test(
                "Non-existent event returns 404",
                response.status_code == 404,
                f"Status: {response.status_code}",
            )

        except Exception as e:
            self.log_test("Non-existent event handling", False, f"Error: {e}")

    async def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "=" * 60)
        print("🚀 ULEI HTTP Server API Test Suite")
        print("=" * 60)

        # Wait for server to be ready
        print("\n⏳ Waiting for server to be ready...")
        for i in range(10):
            try:
                response = await self.client.get(f"{self.base_url}/v1/health")
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    break
            except:
                if i < 9:
                    print(f"  Attempt {i + 1}/10: Server not ready, retrying...")
                    await asyncio.sleep(2)
                else:
                    print("❌ Server not responding after 10 attempts")
                    return

        # Run tests
        await self.test_health_check()
        event_id = await self.test_single_event_ingestion()
        await self.test_batch_ingestion()

        # Give server time to process
        print("\n⏳ Waiting 2 seconds for event processing...")
        await asyncio.sleep(2)

        await self.test_event_status(event_id)  # type: ignore
        await self.test_invalid_requests()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed

        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed / total * 100):.1f}%")

        if failed > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  ❌ {result['name']}: {result['details']}")


async def main():
    """Main test runner."""
    tester = ServerAPITester()
    try:
        await tester.run_all_tests()
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
