#!/usr/bin/env python3
"""
Simple asynchronous load testing script for the Face Recognition API.

This script simulates concurrent users making requests to the /recognize endpoint
and measures performance metrics like requests per second, latency, and error rates.

Dependencies:
- aiohttp
- aiofiles
- numpy
- click
- tqdm

Usage:
  python scripts/load_test.py --url http://localhost:8000/recognize --concurrency 10 --requests 100 --image-path /path/to/test_image.jpg
"""

import asyncio
import time
import base64
import aiohttp
import numpy as np
import click
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def get_image_base64(image_path: str) -> str:
    """Read an image file and encode it as base64."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return None

async def make_request(session: aiohttp.ClientSession, url: str, image_base64: str):
    """Make a single API request and return latency and status."""
    start_time = time.time()
    payload = {
        "image_base64": image_base64,
        "threshold": 0.7,
        "top_k": 5
    }
    
    try:
        async with session.post(url, json=payload) as response:
            latency = time.time() - start_time
            return latency, response.status
    except aiohttp.ClientError as e:
        latency = time.time() - start_time
        logger.warning(f"Request failed: {e}")
        return latency, 999  # Custom error code for client errors

async def worker(session: aiohttp.ClientSession, url: str, image_base64: str, pbar: tqdm):
    """A worker that makes a single request and updates the progress bar."""
    latency, status = await make_request(session, url, image_base64)
    pbar.update(1)
    return latency, status

async def run_load_test(url: str, concurrency: int, num_requests: int, image_path: str):
    """Run the load test with the specified parameters."""
    image_base64 = await get_image_base64(image_path)
    if not image_base64:
        logger.error("Could not load image, aborting load test.")
        return

    latencies = []
    status_codes = []
    
    async with aiohttp.ClientSession() as session:
        with tqdm(total=num_requests, desc="API Load Test", unit="req") as pbar:
            tasks = []
            for _ in range(num_requests):
                task = asyncio.create_task(worker(session, url, image_base64, pbar))
                tasks.append(task)
                
                # Control concurrency
                if len(tasks) >= concurrency:
                    # Wait for one of the running tasks to complete
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for completed_task in done:
                        latency, status = completed_task.result()
                        latencies.append(latency)
                        status_codes.append(status)
                    tasks = list(pending)
            
            # Wait for any remaining tasks
            if tasks:
                done, _ = await asyncio.wait(tasks)
                for completed_task in done:
                    latency, status = completed_task.result()
                    latencies.append(latency)
                    status_codes.append(status)

    return latencies, status_codes

def print_results(latencies: list, status_codes: list, total_time: float):
    """Print the results of the load test."""
    num_requests = len(latencies)
    
    if num_requests == 0:
        logger.info("No requests were completed.")
        return

    # Calculate metrics
    rps = num_requests / total_time
    error_rate = sum(1 for s in status_codes if s >= 400) / num_requests * 100
    
    latencies_np = np.array(latencies)
    avg_latency = np.mean(latencies_np)
    p95_latency = np.percentile(latencies_np, 95)
    p99_latency = np.percentile(latencies_np, 99)
    min_latency = np.min(latencies_np)
    max_latency = np.max(latencies_np)

    # Print summary
    click.echo("\n" + "="*40)
    click.echo("Load Test Results")
    click.echo("="*40)
    click.echo(f"Total Requests:    {num_requests}")
    click.echo(f"Total Time:        {total_time:.2f}s")
    click.echo(f"Requests Per Sec:  {rps:.2f}")
    click.echo(f"Error Rate:        {error_rate:.2f}%")
    click.echo("-"*40)
    click.echo("Latency:")
    click.echo(f"  Average:         {avg_latency:.3f}s")
    click.echo(f"  Min:             {min_latency:.3f}s")
    click.echo(f"  Max:             {max_latency:.3f}s")
    click.echo(f"  95th Percentile: {p95_latency:.3f}s")
    click.echo(f"  99th Percentile: {p99_latency:.3f}s")
    click.echo("="*40)

@click.command()
@click.option('--url', default='http://localhost:8000/recognize', help='API endpoint URL.')
@click.option('--concurrency', '-c', default=10, help='Number of concurrent users.')
@click.option('--requests', '-n', default=100, help='Total number of requests.')
@click.option('--image-path', required=True, type=click.Path(exists=True), help='Path to the test image.')
def main(url: str, concurrency: int, requests: int, image_path: str):
    """A simple async load testing tool for the Face Recognition API."""
    click.echo(f"Starting load test with {concurrency} concurrent users for {requests} requests...")
    
    start_time = time.time()
    latencies, status_codes = asyncio.run(run_load_test(url, concurrency, requests, image_path))
    total_time = time.time() - start_time
    
    print_results(latencies, status_codes, total_time)

if __name__ == "__main__":
    main()
